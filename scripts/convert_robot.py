#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""将包内 Go1 URDF 转换为离线 USD。用法：isaaclab.sh -p scripts/convert_robot.py --headless。"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from pxr import Usd  # noqa: E402
from wmp_lab.reference.assets import configure_reference_colliders, prepare_reference_urdf  # noqa: E402

asset_root = Path(__file__).resolve().parents[1] / "source/wmp_lab/wmp_lab/assets/robots/go1"
root = asset_root / "reference"
try:
    source = prepare_reference_urdf(asset_root / "urdf/go1_isaaclab.urdf", root / "urdf/go1_reference.urdf")
    cfg = UrdfConverterCfg(
        asset_path=str(source),
        usd_dir=str(root),
        usd_file_name="go1.usd",
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        self_collision=True,
        force_usd_conversion=True,
        make_instanceable=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none", gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    )
    converter = UrdfConverter(cfg)
    # URDF 导入器会为无视觉的固定连接生成悬空引用，仅移除这些不存在的引用。
    base = Usd.Stage.Open(str(root / "configuration/go1_base.usd"))
    for prim in base.Traverse():
        references = prim.GetMetadata("references")
        if references:
            for reference in references.GetAddedOrExplicitItems():
                target = root / "configuration" / reference.assetPath if reference.assetPath else None
                stage = Usd.Stage.Open(str(target)) if target and target.is_file() else base
                if reference.primPath and not stage.GetPrimAtPath(reference.primPath).IsValid():
                    prim.GetReferences().RemoveReference(reference)
    base.GetRootLayer().Save()
    count = configure_reference_colliders(root / "go1.usd")
    print(f"[PASS] Offline Go1 USD: {converter.usd_path}; colliders={count}", flush=True)
    go2_root = asset_root.parent / "go2"
    go2_reference = go2_root / "reference/go2.usd"
    go2_reference.parent.mkdir(exist_ok=True)
    if go2_reference.exists():
        go2_reference.unlink()
    source_stage = Usd.Stage.Open(str(go2_root / "go2.usd"))
    stage = Usd.Stage.CreateNew(str(go2_reference))
    prim = stage.DefinePrim(source_stage.GetDefaultPrim().GetPath(), "Xform")
    prim.GetReferences().AddReference("../go2.usd")
    stage.SetDefaultPrim(prim)
    stage.GetRootLayer().Save()
    count = configure_reference_colliders(go2_reference)
    print(f"[PASS] Offline Go2 reference USD: {go2_reference}; colliders={count}", flush=True)
finally:
    app.close()
