#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""通过共享 IsaacLab wrapper 对照原 WarpDepthCamera；固定小场景和相机数量。"""

import argparse
import ast
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
import warp as wp  # noqa: E402
from wmp_lab.sensors.upstream_wmp.warp_cam import WarpCam  # noqa: E402
from wmp_lab.sensors.wmp_warp_camera import WMPWarpDepthCamera  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "wmp_go1"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name in names]
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    scope = dict(torch=torch, np=np, wp=wp, WarpCam=WarpCam)
    definitions(
        Path("/home/re/workspace/isaacgym/python/isaacgym/torch_utils.py"),
        {"quat_apply", "quat_mul", "quat_from_euler_xyz"},
        scope,
    )
    definitions(
        UPSTREAM / "legged_gym/sensors/warp_depth_camera.py", {"WMPDepthCameraConfig", "WarpDepthCamera"}, scope
    )
    pieces = [
        trimesh.creation.box(extents=(8, 8, 0.1), transform=trimesh.transformations.translation_matrix((1, 0, -0.05))),
        trimesh.creation.box(
            extents=(0.13, 0.6, 0.5), transform=trimesh.transformations.translation_matrix((1.05, -0.08, 0.25))
        ),
        trimesh.creation.box(
            extents=(0.45, 0.2, 0.09), transform=trimesh.transformations.translation_matrix((0.65, 0.22, 0.43))
        ),
    ]
    mesh = trimesh.util.concatenate(pieces)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.faces, dtype=np.int32)
    border = 1.5
    reference_vertices = vertices.copy()
    reference_vertices[:, :2] += border
    terrain = SimpleNamespace(vertices=reference_vertices, triangles=triangles, cfg=SimpleNamespace(border_size=border))
    depth_cfg = SimpleNamespace(resized=(64, 64), horizontal_fov=58, near_clip=0, far_clip=2)
    ids = torch.tensor((0, 2, 5, 8), device=args.device)
    n = len(ids)
    angle = torch.tensor((0.03, -0.02, 0.01, -0.01), device=args.device)
    pitch = torch.tensor((15.0, 18.0, 22.0, 25.0), device=args.device) * torch.pi / 180
    local_xyzw = scope["quat_from_euler_xyz"](angle, pitch, angle.flip(0))
    frame = torch.full((n,), -torch.pi / 2, device=args.device)
    optical = scope["quat_from_euler_xyz"](frame, torch.zeros_like(frame), frame)
    ros_wxyz = scope["quat_mul"](local_xyzw, optical)[:, (3, 0, 1, 2)]
    result = dict(passed=False, robot_results={})
    for robot, mount in (("go1", (0.28, 0, 0.12)), ("go2", (0.32715, -0.00003, 0.04297))):
        local_positions = np.repeat(np.asarray(mount, dtype=np.float32)[None], n, axis=0)
        original = scope["WarpDepthCamera"](terrain, depth_cfg, local_positions, local_xyzw.cpu().numpy(), args.device)
        adapted = WMPWarpDepthCamera(
            vertices,
            triangles,
            ids,
            device=args.device,
            local_positions=local_positions,
            local_quaternions_ros=ros_wxyz,
        )
        maximum = 0.0
        for step in range(max(args.steps, 3)):
            root_positions = torch.tensor(
                [[step * 0.02, -0.12, 0.42], [0.1, 0.1, 0.4], [-0.1, 0, 0.39], [0, -0.2, 0.44]], device=args.device
            )
            yaw = angle + step * 0.05
            root_xyzw = scope["quat_from_euler_xyz"](angle * 0.5, angle * 0.3, yaw)
            expected = original.render(root_positions, root_xyzw)
            actual = adapted.render(root_positions, root_xyzw[:, (3, 0, 1, 2)])
            actual = actual[..., 0].clamp(0, 2) / 2 - 0.5
            torch.cuda.synchronize()
            error = float((actual - expected).abs().max())
            maximum = max(maximum, error)
            # 适配层合并固定光学旋转，浮点结合次序可产生数十微米的深度差。
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            assert torch.isfinite(actual).all() and actual.min() >= -0.5 and actual.max() <= 0.5
        report = adapted.buffer_report()
        assert report["image_shape"] == [n, 1, 64, 64]
        assert adapted.camera.graph is not None
        result["robot_results"][robot] = dict(max_abs_error=maximum, **report)
        print(
            f"[PASS] {robot}: original WarpDepthCamera parity, max_abs_error={maximum:.8g}, compact cameras={n}",
            flush=True,
        )
    result["passed"] = True
    output = Path(
        os.environ.get(
            "WMP_CAMERA_VALIDATION_OUTPUT", ROOT / "logs/validation/wmp_camera_go2_20261022/camera_parity.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
