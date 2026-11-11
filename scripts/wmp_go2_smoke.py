#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""旧 Go2 smoke 入口，实际运行统一 Go1/Go2 smoke。"""

import os
import runpy
from pathlib import Path

os.environ.setdefault("WMP_ROBOT", "go2")
runpy.run_path(str(Path(__file__).with_name("wmp_smoke.py")), run_name="__main__")
