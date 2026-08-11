# SPDX-License-Identifier: Apache-2.0
"""Installation metadata for the WMP Go2 IsaacLab extension."""

from pathlib import Path

import toml
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
EXTENSION = toml.load(ROOT / "config" / "extension.toml")

setup(
    name="wmp_go2",
    version=EXTENSION["package"]["version"],
    description=EXTENSION["package"]["description"],
    author=EXTENSION["package"]["author"],
    license="Apache-2.0",
    python_requires=">=3.11",
    packages=find_packages(),
    include_package_data=True,
    package_data={"wmp_go2": ["assets/**/*", "data/**/*"]},
    install_requires=["gymnasium", "numpy", "torch"],
    zip_safe=False,
)
