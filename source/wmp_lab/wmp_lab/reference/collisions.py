# SPDX-License-Identifier: Apache-2.0
"""运行时碰撞修订：不改磁盘资产，只在生成环境时关闭指定连杆碰撞。"""

from pxr import Usd, UsdPhysics

from wmp_lab.robots import (
    COLLISION_REVISION_NATIVE,
    COLLISION_REVISIONS,
    GO2_HEAD_COLLISION_BODIES,
    GO2_NO_HEAD_COLLISION_REVISION,
)


def disable_named_body_collisions(stage, root_path, body_names):
    """关闭 root 下名称匹配的连杆及其子 prim 上的 CollisionAPI。"""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"robot root missing: {root_path}")
    disabled = []
    wanted = set(body_names)
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        parts = {part for part in str(prim.GetPath()).split("/") if part}
        if not (parts & wanted):
            continue
        api = UsdPhysics.CollisionAPI(prim)
        attr = api.GetCollisionEnabledAttr()
        if not attr:
            attr = api.CreateCollisionEnabledAttr(True)
        attr.Set(False)
        disabled.append(str(prim.GetPath()))
    return disabled


def apply_collision_revision(stage, root_path, robot_name, collision_revision):
    """按修订号应用运行时碰撞改动；native 不做任何事。"""
    if collision_revision not in COLLISION_REVISIONS:
        raise ValueError(f"unsupported collision revision {collision_revision!r}")
    if collision_revision == COLLISION_REVISION_NATIVE:
        return []
    if collision_revision == GO2_NO_HEAD_COLLISION_REVISION:
        if robot_name != "go2":
            raise ValueError("no_head_collision is defined only for Go2")
        disabled = disable_named_body_collisions(stage, root_path, GO2_HEAD_COLLISION_BODIES)
        if len(disabled) != 2:
            raise RuntimeError(f"expected 2 Go2 head colliders, disabled={disabled}")
        return disabled
    raise ValueError(f"unhandled collision revision {collision_revision!r}")
