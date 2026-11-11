"""不中断现有作业；其成功完成10000批后，在gpufree续训到累计30000批。"""
from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from gpufree_go2_controller import read, write, rows, process_identity, gpu_report, gpu_idle, source_matches, checkpoint_check

def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()

def remaining_iterations(start, target):
    if not 0 <= start < target:
        raise ValueError("target must exceed the completed iteration")
    return target - start

def previous_ready(state, completion, processes_alive, expected=10000):
    return (state.get("phase") == "completed" and completion.get("exit_code") == 0
            and not completion.get("timed_out", False) and completion.get("iteration") == expected
            and not processes_alive)

def failed_previous(state, completion):
    return state.get("phase") == "failed" or bool(completion and completion.get("exit_code") != 0)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = read(args.config)
    root, work, prior, previous_dir, output = map(Path,
        [cfg["root"], cfg["work"], cfg["prior_work"], cfg["prior_dir"], cfg["output"]])
    lock = (work / "controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(pid=os.getpid(), requested_total_batches=cfg["target"], prior_start_batch=cfg["start"],
                 started_at=now(), phase="initializing", rewards_changed=False, continuation_started=False)
    child = None
    def status(phase, **fields):
        changed = state["phase"] != phase
        state.update(phase=phase, updated_at=now(), **fields)
        write(work / "state.json", state)
        if changed:
            print(json.dumps(state, ensure_ascii=False), flush=True)
    def unchanged():
        if not source_matches(root, cfg["source_hashes"]) or not source_matches(root, cfg["execution_hashes"]):
            raise RuntimeError("approved source or launcher changed; refusing continuation")
    try:
        if (work / "launch_continuation.json").exists() or (output / "run.json").exists():
            raise RuntimeError("continuation already launched; refusing duplicate")
        while True:
            unchanged()
            old_state = read(prior / "state.json", {})
            complete = read(previous_dir / "job_completion.json", {})
            if failed_previous(old_state, complete):
                raise RuntimeError("original training failed; continuation is not started automatically")
            alive = any(process_identity(int(pid)) == identity for pid, identity in cfg["wait_processes"].items())
            if previous_ready(old_state, complete, alive, cfg["start"]):
                break
            status("waiting_current_10000", current_training_alive=alive, prior_phase=old_state.get("phase"),
                   target_total=cfg["target"], continuation_directory=str(output))
            time.sleep(20)
        checkpoint = previous_dir / f"model_{cfg['start']}.pt"
        audit = checkpoint_check(cfg["python"], checkpoint)
        if audit["iteration"] != cfg["start"] or audit["total_steps"] != cfg["start"] * 98304:
            raise RuntimeError("previous final checkpoint has the wrong iteration/sample count")
        write(work / "input_checkpoint_audit.json", dict(path=str(checkpoint), **audit))
        stable = 0
        while stable < 3:
            unchanged()
            gpu = gpu_report()
            stable = stable + 1 if gpu_idle(gpu) else 0
            status("waiting_gpu", gpu=gpu, idle_checks=stable)
            if stable < 3:
                time.sleep(20)
        output.mkdir(parents=True, exist_ok=False)
        script = output / "run_climb_training.py"
        script.write_bytes(Path(cfg["probe_template"]).read_bytes())
        env = dict(os.environ, WMP_PROJECT_ROOT=str(root), WMP_SHORT_NUM_ENVS="4096",
                   WMP_SHORT_CAMERA_ENVS="1100", WMP_SHORT_TRAIN_DIR=str(output),
                   WMP_TRAIN_CHECKPOINT=str(checkpoint), WMP_TRAIN_SAVE_INTERVAL="500",
                   CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
        additional = remaining_iterations(cfg["start"], cfg["target"])
        argv = ["/bin/bash", str(root / "bash/gpufree_isaaclab.sh"), str(script),
                "--headless", "--device", "cuda:0", "--steps", str(additional)]
        with (output / "console.log").open("ab", buffering=0) as stream:
            child = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        launch = dict(pid=child.pid, argv=argv, from_scratch=False, start_iteration=cfg["start"],
                      target_iteration=cfg["target"], additional_iterations=additional,
                      resume_checkpoint=str(checkpoint), save_interval=500, timeout_s=432000,
                      num_envs=4096, camera_envs=1100, rewards_changed=False, started_at=now())
        write(output / "launch.json", launch)
        write(work / "launch_continuation.json", dict(directory=str(output), **launch))
        status("continuation_running", continuation_started=True, child_pid=child.pid, directory=str(output))
        started = time.monotonic()
        first_seen, timeout = False, False
        while child.poll() is None:
            if not first_seen:
                data = rows(output / "metrics.jsonl")
                if data:
                    if data[0]["iteration"] != cfg["start"] + 1 or data[0]["total_steps"] != (cfg["start"] + 1) * 98304:
                        os.killpg(child.pid, signal.SIGTERM)
                        raise RuntimeError("continuation did not resume at the expected iteration")
                    write(output / "start_confirmation.json", dict(confirmed_at=now(), pid=child.pid,
                        first_iteration=data[0]["iteration"], target_iteration=cfg["target"]))
                    first_seen = True
            if time.monotonic() - started > 432000:
                timeout = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(5)
        code = child.wait()
        data = rows(output / "metrics.jsonl")
        result = dict(exit_code=code, timed_out=timeout, finished_at=now(),
                      iteration=data[-1]["iteration"] if data else cfg["start"],
                      total_steps=data[-1]["total_steps"] if data else cfg["start"]*98304,
                      target_iteration=cfg["target"], additional_batches_logged=len(data))
        write(output / "job_completion.json", result)
        if code or timeout or len(data) != additional or result["iteration"] != cfg["target"]:
            raise RuntimeError("continuation did not finish its requested bound")
        final = checkpoint_check(cfg["python"], output / f"model_{cfg['target']}.pt")
        unchanged()
        write(work / "completion.json", dict(status="completed", checkpoint_audit=final, **result))
        status("completed", final_iteration=cfg["target"])
        return 0
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        write(work / "failure.json", dict(error=str(error), failed_at=now(), state=state))
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
