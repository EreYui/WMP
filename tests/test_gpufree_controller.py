"""验证定时门禁、完整批次门禁和容器内存语义；不会启动训练。"""
import importlib.util
from pathlib import Path
import sys
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from gpufree_go2_controller import start_gate, validate_metrics, source_matches
from resource_memory import memory_report, MIB

def test_launch_requires_time_validation_and_idle():
    assert not start_gate(99, 100, True, True)
    assert not start_gate(100, 100, False, True)
    assert not start_gate(100, 100, True, False)
    assert start_gate(100, 100, True, True)
    assert start_gate(150, 100, True, True)

def test_container_memory_includes_only_reclaimable_inactive_cache(tmp_path):
    (tmp_path / "memory.max").write_text(str(50 * MIB))
    (tmp_path / "memory.current").write_text(str(48 * MIB))
    (tmp_path / "memory.stat").write_text(f"inactive_file {20 * MIB}\nactive_file {10 * MIB}\n")
    r = memory_report(tmp_path, 500 * MIB)
    assert r["host_available_mib"] == 22
    assert r["cgroup_working_set_mib"] == 28
    assert memory_report(tmp_path, 10 * MIB)["host_available_mib"] == 10

def test_no_cgroup_falls_back_to_host(tmp_path):
    assert memory_report(tmp_path, 5 * MIB)["host_available_mib"] == 5

def good_rows():
    return [dict(iteration=i, total_steps=i*98304, batch_samples=98304, collection_steps=24,
                 gae_horizon=24, ppo_minibatch_samples=24576, ppo_optimizer_steps=20,
                 curriculum_reference_iteration=i-1, world_optimizer_steps=10, depth_optimizer_steps=50,
                 world_sequence_length=64, gpu_peak_allocated_mib=5000, host_available_mib=10000)
            for i in range(1, 6)]

def test_validation_rejects_partial_and_nonfinite_batches():
    r = good_rows()
    assert validate_metrics(r, 5)["samples"] == 5 * 98304
    with pytest.raises(ValueError): validate_metrics(r[:-1], 5)
    r[0]["extra"] = float("nan")
    with pytest.raises(ValueError): validate_metrics(r, 5)

def test_validation_requires_full_world_model_and_depth_updates():
    r = good_rows()
    r[-1]["world_optimizer_steps"] = 0
    with pytest.raises(ValueError): validate_metrics(r, 5)
    r = good_rows()
    for row in r:row["world_sequence_length"] = 4
    with pytest.raises(ValueError): validate_metrics(r, 5)

def test_source_change_blocks_startup(tmp_path):
    import hashlib
    f = tmp_path / "robot.py"
    f.write_text("before")
    hashes = {"robot.py":hashlib.sha256(f.read_bytes()).hexdigest()}
    assert source_matches(tmp_path, hashes)
    f.write_text("after")
    assert not source_matches(tmp_path, hashes)

def test_controller_entire_success_path_uses_fresh_validation_then_long(tmp_path, monkeypatch):
    import gpufree_go2_controller as module
    work = tmp_path / "work"
    work.mkdir()
    module.write(work / "preflight.json", {"passed":True})
    validation = tmp_path / "validation"
    validation.mkdir()
    module.write(validation / "runtime_audit.json", {"passed":True})
    module.write(validation / "run.json", {"args":{"resume":"False","checkpoint":"None"},
                                         "learning":{"num_envs":4096,"collection_chunks":1}})
    config = {"root":str(tmp_path), "work":str(work), "earliest_start":"1970-01-01T00:00:00+00:00",
              "source_hashes":{}, "validation_dir":str(validation), "long_dir":str(tmp_path / "long"),
              "validation_batches":5, "long_batches":5, "python":"unused"}
    controller = module.Controller(config)
    monkeypatch.setattr(controller, "wait_resources", lambda phase: {"free_mib":23000,"utilization":0})
    monkeypatch.setattr(module, "checkpoint_check", lambda *args: {"iteration":5,"total_steps":491520})
    calls = []
    def fake_child(label, output, batches, save_interval, timeout_s):
        calls.append((label,batches))
        return good_rows(), {"exit_code":0,"iteration":5,"total_steps":491520,
                             "checkpoint":str(tmp_path / "model_5.pt")}
    monkeypatch.setattr(controller, "run_child", fake_child)
    controller.run()
    assert calls == [("validation",5),("long",5)]
    assert module.read(work / "validation_passed.json")["passed"]
    assert module.read(work / "completion.json")["status"] == "completed"

def test_failed_validation_never_launches_long(tmp_path, monkeypatch):
    import gpufree_go2_controller as module
    work = tmp_path / "work"
    work.mkdir()
    module.write(work / "preflight.json", {"passed":True})
    config = {"root":str(tmp_path), "work":str(work), "earliest_start":"1970-01-01T00:00:00+00:00",
              "source_hashes":{}, "validation_dir":str(tmp_path/"validation"),
              "validation_batches":5}
    controller = module.Controller(config)
    monkeypatch.setattr(controller, "wait_resources", lambda phase: {})
    calls=[]
    def fake_child(label,*args):
        calls.append(label)
        return good_rows()[:-1], {}
    monkeypatch.setattr(controller,"run_child",fake_child)
    with pytest.raises(ValueError):controller.run()
    assert calls==["validation"]
    assert not (work/"launch_long.json").exists()
