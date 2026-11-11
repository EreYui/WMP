# SPDX-License-Identifier: Apache-2.0
"""不启动 Gym，执行原采样函数核对时间分布、插值、预加载及续训契约。"""

import ast
import json
import shutil
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from wmp_lab.learning.amp import AMPMotionDataset
from wmp_lab.learning.config import SMOKE_CFG
from wmp_lab.learning.runner import WMPRunner

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source/wmp_lab/wmp_lab/data/motions_upstream"


def original_sampler(seed):
    path = ROOT.parent / "wmp_go1/rsl_rl/datasets/motion_loader.py"
    cls = next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "AMPLoader"
    )
    selected = {
        "weighted_traj_idx_sample_batch",
        "traj_time_sample_batch",
        "slerp",
        "get_frame_at_time_batch",
        "feed_forward_generator",
    }
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    rng = np.random.RandomState(seed)
    # np.int 在新版 NumPy 已移除；只替换这个类型别名，不修改原函数数学定义。
    scope = dict(
        torch=torch,
        np=SimpleNamespace(**dict(np.__dict__, int=np.int64, random=rng)),
        AMPLoader=SimpleNamespace(
            JOINT_POSE_START_IDX=7, JOINT_POSE_END_IDX=19, LINEAR_VEL_START_IDX=31, JOINT_VEL_END_IDX=49
        ),
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
    manifest = json.loads((DATA / "manifest.json").read_text())
    raw = [json.loads((ROOT / clip["source"]).read_text()) for clip in manifest["sequences"]]
    frames = [torch.tensor(clip["Frames"], dtype=torch.float32)[:, :49] for clip in raw]
    durations = np.asarray([clip["FrameDuration"] for clip in raw])
    sizes = np.asarray([len(value) for value in frames], dtype=float)
    weights = np.asarray([clip["MotionWeight"] for clip in raw], dtype=float)
    fake = SimpleNamespace(
        trajectories=frames,
        observation_dim=49,
        trajectory_idxs=list(range(len(frames))),
        trajectory_weights=weights / weights.sum(),
        trajectory_lens=(sizes - 1) * durations,
        trajectory_frame_durations=durations,
        trajectory_num_frames=sizes,
        time_between_frames=0.02,
        device="cpu",
        preload_transitions=True,
    )
    for name in selected:
        setattr(fake, name, MethodType(scope[name], fake))
    return fake


def amp_columns(full):
    return torch.cat((full[:, 7:19], full[:, 31:49]), -1)


@pytest.mark.parametrize("preload", [0, 257])
def test_amp_pairs_match_original_timing_interpolation_and_preloaded_feed(preload):
    seed, count = 17, 1024
    ref = original_sampler(seed)
    dataset = AMPMotionDataset(DATA, seed=seed, sampling_mode="upstream_v1", preload=preload)
    ids = ref.weighted_traj_idx_sample_batch(preload or count)
    times = ref.traj_time_sample_batch(ids)
    # 原函数在零点具有概率质量，并非在合法区间重新均匀采样。
    assert np.count_nonzero(times == 0) > 0
    first = ref.get_frame_at_time_batch(ids, times)
    second = ref.get_frame_at_time_batch(ids, times + 0.02)
    if preload:
        torch.testing.assert_close(dataset.preloaded[0], amp_columns(first), rtol=0, atol=0)
        torch.testing.assert_close(dataset.preloaded[1], amp_columns(second), rtol=0, atol=0)
        ref.preloaded_s, ref.preloaded_s_next = first, second
        expected = next(ref.feed_forward_generator(1, count))
    else:
        expected = amp_columns(first), amp_columns(second)
    actual = dataset.sample_transitions(count)
    for ours, original in zip(actual[:2], expected, strict=True):
        torch.testing.assert_close(ours, original, rtol=0, atol=0)


def test_amp_rng_and_contract_survive_checkpoint_and_reject_other_data(tmp_path):
    expert = AMPMotionDataset(DATA, seed=7, sampling_mode="upstream_v1", preload=64)
    runner = WMPRunner(SMOKE_CFG)
    runner.bind_expert(expert)
    expert.sample_transitions(37)
    checkpoint = runner.save(tmp_path / "model.pt")
    expected = expert.sample_transitions(31)
    clone = WMPRunner.from_checkpoint(checkpoint)
    resumed = AMPMotionDataset(DATA, seed=7, sampling_mode="upstream_v1", preload=64)
    clone.bind_expert(resumed)
    for actual, value in zip(resumed.sample_transitions(31), expected, strict=True):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    wrong = AMPMotionDataset(DATA, seed=8, sampling_mode="upstream_v1", preload=64)
    with pytest.raises(ValueError, match="contract differs"):
        clone.bind_expert(wrong)
    with pytest.raises(ValueError, match="dt differs"):
        resumed.sample_transitions(1, dt=0.01)
    payload = torch.load(checkpoint, weights_only=False)
    payload["extra"].pop("amp_dataset")
    payload["iteration"] = 1
    torch.save(payload, tmp_path / "no_contract.pt")
    historical = WMPRunner.from_checkpoint(tmp_path / "no_contract.pt")
    with pytest.raises(ValueError, match="missing its AMP contract"):
        historical.bind_expert(resumed)


def test_contract_allows_relocated_files_but_rejects_changed_clip(tmp_path):
    original = AMPMotionDataset(DATA, sampling_mode="upstream_v1")
    manifest = json.loads((DATA / "manifest.json").read_text())
    shutil.copy2(DATA / "manifest.json", tmp_path / "manifest.json")
    for entry in manifest["sequences"]:
        shutil.copy2(DATA / entry["file"], tmp_path / entry["file"])
    moved = AMPMotionDataset(tmp_path, sampling_mode="upstream_v1")
    assert moved.contract["fingerprint"] == original.contract["fingerprint"]
    path = tmp_path / manifest["sequences"][0]["file"]
    with np.load(path) as values:
        arrays = {key: values[key] for key in values.files}
    arrays["joint_pos"][0, 0] += 0.1
    np.savez(path, **arrays)
    changed = AMPMotionDataset(tmp_path, sampling_mode="upstream_v1")
    runner = WMPRunner(SMOKE_CFG)
    runner.bind_expert(original)
    with pytest.raises(ValueError, match="contract differs"):
        runner.bind_expert(changed)
