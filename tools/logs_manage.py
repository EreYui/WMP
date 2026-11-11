#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""预览/压缩日志或清理旧模型。用法：python tools/logs_manage.py {compress|delete} RUN [--y]。"""

import argparse
import re
import tarfile
from pathlib import Path


def candidates(root: Path, keep: int):
    files = list(root.glob("model_*.pt"))

    def iteration(path):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        return int(match.group(1)) if match else -1

    files = sorted((path for path in files if iteration(path) >= 0), key=iteration)
    return files[:-keep] if keep else files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("compress", "delete"))
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--y", action="store_true", help="Execute the previewed operation")
    args = parser.parse_args()
    if args.keep < 0:
        parser.error("keep must be non-negative")
    for root in args.runs:
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"not a run directory: {root}")
        if args.operation == "delete":
            for path in candidates(root, args.keep):
                print(f"{'DELETE' if args.y else 'PREVIEW DELETE'} {path}")
                if args.y and not path.is_symlink():
                    path.unlink()
        else:
            output = root.with_name(root.name + ".tar.gz")
            print(f"{'COMPRESS' if args.y else 'PREVIEW COMPRESS'} {root} -> {output}")
            if args.y:
                if output.exists():
                    raise FileExistsError(output)
                with tarfile.open(output, "w:gz", dereference=False) as archive:
                    archive.add(root, arcname=root.name)


if __name__ == "__main__":
    main()
