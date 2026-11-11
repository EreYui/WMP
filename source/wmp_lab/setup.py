# SPDX-License-Identifier: Apache-2.0
"""Go1/Go2 WMP IsaacLab 扩展安装元数据。"""

from pathlib import Path

import toml
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
EXTENSION = toml.load(ROOT / "config" / "extension.toml")

setup(
    name="wmp_lab",
    version=EXTENSION["package"]["version"],
    description=EXTENSION["package"]["description"],
    author=EXTENSION["package"]["author"],
    license="Apache-2.0",
    python_requires=">=3.11",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "wmp_lab": [
            "assets/**/*",
            "data/**/*",
            "learning/upstream_dreamer/*.yaml",
            "learning/**/LICENSE*",
            "learning/*LICENSE*",
            "sensors/upstream_wmp/*.json",
        ]
    },
    install_requires=["gymnasium", "numpy", "scipy", "trimesh", "torch", "tensorboard"],
    extras_require={
        "tools": ["tensorboard", "matplotlib", "opencv-python", "imageio[ffmpeg]"],
        "mujoco": ["mujoco>=3.3"],
    },
    zip_safe=False,
)
