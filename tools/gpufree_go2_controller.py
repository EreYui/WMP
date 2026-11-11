"""目标机驻留控制器：等旧作业结束、完整规模验证通过，最早北京时间10点从零长训。"""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from resource_memory import memory_report

TZ = ZoneInfo("Asia/Shanghai")

def now():
    return datetime.now(TZ).isoformat()

def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default

def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)

def rows(path):
    if not Path(path).exists():
        return []
    result = []
    with Path(path).open() as stream:
        for line in stream:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result

def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if text[0] == "Z" else text[19]
    except OSError:
        return None

def start_gate(current, earliest, validation_passed, gpu_idle):
    return current >= earliest and validation_passed and gpu_idle

def source_matches(root, hashes):
    return all((root / path).is_file() and hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
               for path, digest in hashes.items())

def gpu_report():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            timeout=15, check=True)
    total, used, utilization = map(float, result.stdout.splitlines()[0].split(","))
    return {"total_mib": total, "used_mib": used, "free_mib": total - used, "utilization": utilization}

def gpu_idle(report):
    return report["free_mib"] >= 22000 and report["utilization"] <= 15

def validate_metrics(data, expected):
    if len(data) != expected or [r["iteration"] for r in data] != list(range(1, expected + 1)):
        raise ValueError("validation did not finish the requested fresh batches")
    for r in data:
        if any(not math.isfinite(v) for v in r.values() if isinstance(v, (float, int))):
            raise ValueError("nonfinite validation metric")
        required = {"batch_samples": 98304, "collection_steps": 24, "gae_horizon": 24,
                    "ppo_minibatch_samples": 24576, "ppo_optimizer_steps": 20}
        if any(r.get(k) != v for k, v in required.items()):
            raise ValueError("4096 reference collection/PPO mismatch")
        if r["total_steps"] != r["iteration"] * 98304:
            raise ValueError("sample clock mismatch")
        if r["curriculum_reference_iteration"] != r["iteration"] - 1:
            raise ValueError("curriculum clock mismatch")
    mature = [r for r in data if r.get("world_optimizer_steps") == 10 and r.get("depth_optimizer_steps") == 50]
    if len(mature) < 5 or any(r.get("world_optimizer_steps") != 10 or r.get("depth_optimizer_steps") != 50
                              for r in data[-5:]):
        raise ValueError("world/depth updates did not warm up to 10/50")
    if max(r.get("world_sequence_length", 0) for r in mature) < 64:
        raise ValueError("full 64-frame world-model update not observed")
    return {"batches": expected, "samples": expected * 98304, "mature_updates": len(mature),
            "max_world_sequence_length": max(r.get("world_sequence_length", 0) for r in mature),
            "peak_torch_allocated_mib": max(r["gpu_peak_allocated_mib"] for r in data),
            "minimum_effective_host_available_mib": min(r["host_available_mib"] for r in data)}

def checkpoint_check(python, path):
    code = """import json,torch,sys
p=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
def ts(v):
 if isinstance(v,torch.Tensor):yield v
 elif isinstance(v,dict):
  for x in v.values():yield from ts(x)
 elif isinstance(v,(list,tuple)):
  for x in v:yield from ts(x)
bad=sum(not bool(torch.isfinite(x).all()) for x in ts(p) if x.is_floating_point())
assert bad==0
print(json.dumps(dict(iteration=p['iteration'],total_steps=p['extra']['total_steps'],nonfinite_tensors=bad)))
"""
    q = subprocess.run([python, "-B", "-c", code, str(path)], capture_output=True, text=True, timeout=180)
    if q.returncode:
        raise RuntimeError("checkpoint verification failed: " + q.stderr[-2000:])
    return json.loads(q.stdout.strip().splitlines()[-1])

class Controller:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["root"])
        self.work = Path(config["work"])
        self.state = dict(controller_pid=os.getpid(), started_at=now(), phase="initializing",
                          requested_start_at=config["earliest_start"], training_started=False,
                          validation_passed=False, rewards_changed=False)
        self.lock = (self.work / "controller.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.work / "launch_long.json").exists():
            raise RuntimeError("long training was already launched; refusing duplicate startup")

    def status(self, phase, **fields):
        changed = phase != self.state["phase"]
        self.state.update(phase=phase, updated_at=now(), **fields)
        write(self.work / "state.json", self.state)
        if changed:
            print(json.dumps(self.state, ensure_ascii=False), flush=True)

    def healthy_source(self):
        if not source_matches(self.root, self.config["source_hashes"]):
            raise RuntimeError("production source changed after preflight; refusing startup")
        if not source_matches(self.root, self.config.get("execution_hashes", {})):
            raise RuntimeError("approved launcher/monitor code changed; refusing startup")

    def wait_resources(self, phase):
        stable = 0
        while True:
            self.healthy_source()
            active = process_identity(self.config["wait_pid"]) == self.config["wait_start_ticks"]
            report = gpu_report()
            memory = memory_report()
            stable = stable + 1 if not active and gpu_idle(report) and memory["host_available_mib"] >= 8000 else 0
            self.status(phase, prior_job_alive=active, gpu=report, memory=memory, idle_checks=stable)
            if stable >= 3:
                return report
            time.sleep(self.config["poll_seconds"])

    def run_child(self, label, output, batches, save_interval, timeout_s):
        self.healthy_source()
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if (output / "launch.json").exists() or (output / "run.json").exists():
            raise RuntimeError("output already contains a run")
        script = output / "run_climb_training.py"
        shutil.copy2(self.config["probe_template"], script)
        env = dict(os.environ)
        env.pop("WMP_TRAIN_CHECKPOINT", None)
        env.update(WMP_PROJECT_ROOT=str(self.root), WMP_SHORT_NUM_ENVS="4096", WMP_SHORT_CAMERA_ENVS="1100",
                   WMP_SHORT_TRAIN_DIR=str(output), WMP_TRAIN_SAVE_INTERVAL=str(save_interval),
                   CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
        args = ["/bin/bash", str(self.root / "bash/gpufree_isaaclab.sh"), str(script),
                "--headless", "--device", "cuda:0", "--steps", str(batches)]
        with (output / "console.log").open("ab", buffering=0) as stream:
            proc = subprocess.Popen(args, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        launch = dict(pid=proc.pid, argv=args, started_at=now(), from_scratch=True, checkpoint=None,
                      num_envs=4096, camera_envs=1100, batches=batches, timeout_s=timeout_s,
                      save_interval=save_interval, rewards_changed=False)
        write(output / "launch.json", launch)
        if label == "long":
            write(self.work / "launch_long.json", dict(directory=str(output), **launch))
        start = time.monotonic()
        self.status(label + "_running", child_pid=proc.pid, directory=str(output),
                    training_started=label == "long")
        first_seen = False
        timed_out = False
        while proc.poll() is None:
            if label == "long" and not first_seen:
                data = rows(output / "metrics.jsonl")
                if data:
                    first = data[0]
                    if first["iteration"] != 1 or first["total_steps"] != 98304:
                        raise RuntimeError("fresh long startup did not start at batch 1")
                    write(output / "start_confirmation.json", dict(confirmed_at=now(), pid=proc.pid,
                                                                  first_batch=first, from_scratch=True))
                    first_seen = True
            if time.monotonic() - start > timeout_s:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(5)
        code = proc.wait()
        data = rows(output / "metrics.jsonl")
        resources = rows(output / "resources.jsonl")
        result = dict(exit_code=code, timed_out=timed_out, finished_at=now(),
                      iteration=data[-1]["iteration"] if data else 0,
                      total_steps=data[-1]["total_steps"] if data else 0,
                      peak_device_mib=max((r["gpu_used_mib"] for r in resources), default=None),
                      effective_memory_min_mib=min((r["host_available_mib"] for r in resources), default=None),
                      checkpoint=str(output / f"model_{data[-1]['iteration']}.pt") if data else None)
        write(output / "job_completion.json", result)
        if code or timed_out:
            raise RuntimeError(f"{label} exited with code {code}; see {output}/console.log")
        return data, result

    def run(self):
        preflight = read(self.work / "preflight.json", {})
        if not preflight.get("passed"):
            raise RuntimeError("CPU/environment preflight is incomplete")
        self.wait_resources("waiting_existing_training_and_gpu")
        data, result = self.run_child("validation", self.config["validation_dir"],
                                      self.config["validation_batches"], 50, 5400)
        checked = validate_metrics(data, self.config["validation_batches"])
        output = Path(self.config["validation_dir"])
        audit = read(output / "runtime_audit.json", {})
        if not audit.get("passed"):
            raise RuntimeError("runtime robot/camera audit did not pass")
        run = read(output / "run.json", {})
        if run.get("args", {}).get("resume") != "False" or run["args"].get("checkpoint") != "None":
            raise RuntimeError("validation unexpectedly resumed an old checkpoint")
        if run["learning"]["num_envs"] != 4096 or run["learning"]["collection_chunks"] != 1:
            raise RuntimeError("validation used the wrong physical environment count")
        ckpt = checkpoint_check(self.config["python"], output / f"model_{self.config['validation_batches']}.pt")
        checked.update(result)
        checked.update(passed=True, checkpoint_audit=ckpt, runtime_audit=audit, completed_at=now())
        write(self.work / "validation_passed.json", checked)
        self.state["validation_passed"] = True
        earliest = datetime.fromisoformat(self.config["earliest_start"]).timestamp()
        while time.time() < earliest:
            self.status("waiting_beijing_1000", validation_passed=True)
            time.sleep(min(30, max(0.1, earliest - time.time())))
        report = self.wait_resources("waiting_gpu_before_long")
        if not start_gate(time.time(), earliest, True, gpu_idle(report)):
            raise RuntimeError("long startup gate failed")
        self.healthy_source()
        data, result = self.run_child("long", self.config["long_dir"], self.config["long_batches"],
                                      500, 259200)
        if len(data) != self.config["long_batches"] or data[-1]["iteration"] != self.config["long_batches"]:
            raise RuntimeError("long run stopped before its requested bound")
        final = checkpoint_check(self.config["python"], Path(result["checkpoint"]))
        write(self.work / "completion.json", dict(status="completed", checkpoint_audit=final, **result))
        self.status("completed", final_iteration=final["iteration"])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    controller = Controller(read(args.config))
    try:
        controller.run()
    except BaseException as error:
        controller.status("failed", error=f"{type(error).__name__}: {error}")
        write(controller.work / "failure.json", dict(error=str(error), failed_at=now(),
                                                    state=controller.state))
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
