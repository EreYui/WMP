# SPDX-License-Identifier: Apache-2.0
"""wmp_go1 Warp 相机的 IsaacLab 适配，仅为选中的环境分配相机缓冲。"""

from types import SimpleNamespace

import torch
import warp as wp

from .upstream_wmp.warp_cam import WarpCam


@torch.jit.script
def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


@torch.jit.script
def quat_apply_wxyz(q: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    uv = torch.cross(q[..., 1:], value, dim=-1)
    return value + 2 * (q[..., :1] * uv + torch.cross(q[..., 1:], uv, dim=-1))


class WMPWarpDepthCamera:
    """复用原 WarpCam/射线 kernel；输入机身wxyz，内部仅转为Warp所需xyzw。

    local_quaternions_ros是相机光学坐标系的安装旋转，已包含原frame_quat。
    render返回米制轴向深度；归一化和跨帧延迟仍由环境处理。
    """

    implementation = "wmp_go1_warp_v1"

    def __init__(
        self,
        vertices,
        triangles,
        camera_env_ids,
        *,
        device,
        local_positions,
        local_quaternions_ros,
        width=64,
        height=64,
        horizontal_fov=58.0,
        far_clip=2.0,
    ):
        wp.init()
        self.device = str(device)
        self.camera_env_ids = camera_env_ids.to(device=device, dtype=torch.long).clone()
        self.num_cameras = len(self.camera_env_ids)
        if self.num_cameras == 0:
            raise ValueError("compact camera requires at least one selected environment")
        self.width, self.height = width, height
        self.mesh_points = torch.as_tensor(vertices, device=device, dtype=torch.float32).contiguous()
        self.mesh_indices = torch.as_tensor(triangles, device=device, dtype=torch.int32).flatten().contiguous()
        self.mesh = wp.Mesh(
            points=wp.from_torch(self.mesh_points, dtype=wp.vec3),
            indices=wp.from_torch(self.mesh_indices, dtype=wp.int32),
        )
        self.mesh_ids = wp.array([self.mesh.id] * self.num_cameras, dtype=wp.uint64, device=self.device)
        cfg = SimpleNamespace(
            num_sensors=1,
            width=width,
            height=height,
            horizontal_fov_deg=horizontal_fov,
            max_range=far_clip,
            calculate_depth=True,
            return_pointcloud=False,
            segmentation_camera=False,
            pointcloud_in_world_frame=False,
        )
        self.camera = WarpCam(self.num_cameras, cfg, self.mesh_ids, device=self.device)
        self.pixels = torch.zeros((self.num_cameras, 1, height, width), device=device)
        self.camera_positions = torch.zeros((self.num_cameras, 1, 3), device=device)
        self.camera_orientations = torch.zeros((self.num_cameras, 1, 4), device=device)
        self.camera_orientations[..., 3] = 1
        self.local_positions = torch.zeros((self.num_cameras, 3), device=device)
        self.local_quaternions_ros = torch.zeros((self.num_cameras, 4), device=device)
        self.set_local_poses(local_positions, local_quaternions_ros)
        self.camera.set_pose_tensor(self.camera_positions, self.camera_orientations)
        self.camera.set_image_tensors(self.pixels)
        self._warmed = False
        self._render_stream = wp.Stream(device=self.device) if self.pixels.is_cuda else None

    def set_local_poses(self, positions, quaternions_ros):
        self.local_positions.copy_(torch.as_tensor(positions, device=self.device).expand(self.num_cameras, 3))
        self.local_quaternions_ros.copy_(
            torch.as_tensor(quaternions_ros, device=self.device).expand(self.num_cameras, 4)
        )

    @torch.no_grad()
    def render(self, root_positions, root_quaternions_wxyz):
        if root_positions.shape != (self.num_cameras, 3) or root_quaternions_wxyz.shape != (self.num_cameras, 4):
            raise ValueError("camera poses must follow compact camera_env_ids order")
        positions = quat_apply_wxyz(root_quaternions_wxyz, self.local_positions) + root_positions
        orientations = quat_mul_wxyz(root_quaternions_wxyz, self.local_quaternions_ros)
        self.camera_positions[:, 0].copy_(positions)
        self.camera_orientations[:, 0].copy_(orientations[:, (1, 2, 3, 0)])
        if root_positions.is_cuda:
            # Kit内置Warp不能在外部Torch默认流上捕获图；用自有流和事件双向排序。
            torch_stream = wp.stream_from_torch(torch.cuda.current_stream(root_positions.device))
            self._render_stream.wait_stream(torch_stream)
            with wp.ScopedStream(self._render_stream):
                if not self._warmed:
                    self.camera.capture(debug=True)
                    self._warmed = True
                self.camera.capture()
            torch_stream.wait_stream(self._render_stream)
        else:
            self.camera.capture(debug=True)
        return self.pixels[:, 0, :, :, None]

    def buffer_report(self):
        camera_tensors = (
            self.pixels,
            self.camera_positions,
            self.camera_orientations,
            self.local_positions,
            self.local_quaternions_ros,
        )
        return dict(
            implementation=self.implementation,
            cameras=self.num_cameras,
            image_shape=list(self.pixels.shape),
            camera_tensor_bytes=sum(x.numel() * x.element_size() for x in camera_tensors),
            shared_mesh_tensor_bytes=self.mesh_points.numel() * self.mesh_points.element_size()
            + self.mesh_indices.numel() * self.mesh_indices.element_size(),
        )
