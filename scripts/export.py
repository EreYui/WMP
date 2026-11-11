#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""导出 Go1/Go2 有状态策略。用法：python scripts/export.py --checkpoint model.pt --output policy.pt。"""

import argparse

from wmp_lab.learning.deployment import export_policy
from wmp_lab.learning.runner import WMPRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    runner = WMPRunner.from_checkpoint(args.checkpoint)
    path, error = export_policy(runner, args.output, args.batch_size)
    print(f"Exported {path}; max_abs_error={error:.6g}")


if __name__ == "__main__":
    main()
