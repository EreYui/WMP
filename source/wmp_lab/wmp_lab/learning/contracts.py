# SPDX-License-Identifier: Apache-2.0
"""保存完整环境、源码和资产契约，禁止在续训时静默改变实验。"""

import hashlib
import json
from pathlib import Path


def canonical(value):
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list | tuple):
        return [canonical(item) for item in value]
    if isinstance(value, slice):
        return dict(slice=[value.start, value.stop, value.step])
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        target = value if hasattr(value, "__qualname__") else type(value)
        return f"{target.__module__}.{target.__qualname__}"
    if hasattr(value, "to_dict"):
        return canonical(value.to_dict())
    return str(value)


def environment_contract(cfg):
    package = Path(__file__).resolve().parents[1]
    source = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
        if "assets" not in path.parts
    }
    asset_root = package / "assets" / "robots" / cfg.robot_name
    assets = {
        str(path.relative_to(asset_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(asset_root.rglob("*"))
        if path.suffix in (".usd", ".usda", ".usdc", ".urdf")
    }
    payload = dict(environment_revision=cfg.environment_revision, config=canonical(cfg), source=source, assets=assets)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(fingerprint=digest, **payload)
