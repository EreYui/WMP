#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""查看全地形和策略。用法：isaaclab.sh -p scripts/view_terrain.py --checkpoint model.pt。"""

import runpy
import sys
from pathlib import Path

sys.argv.extend(["--terrain-view"])
if not any(arg == "--num-envs" or arg.startswith("--num-envs=") for arg in sys.argv):
    sys.argv.extend(["--num-envs", "18"])
runpy.run_path(str(Path(__file__).with_name("play.py")), run_name="__main__")
