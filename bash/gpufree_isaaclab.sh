#!/usr/bin/env bash
# gpufree 专用入口：使用已有环境及 wheel 内的完整 IsaacLab core，不改共享依赖。
set -euo pipefail
source /root/gpufree-data/yfx/activate_isaaclab.sh
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WMP_PROJECT_ROOT="$root"
export ISAACLAB_ROOT=/root/gpufree-data/yfx/code/isaaclab
runtime=/root/gpufree-data/yfx/envs/isaaclab/lib/python3.11/site-packages/isaaclab
export ISAACLAB_RUNTIME_ROOT="$runtime"
export TERM=xterm-256color
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/source/wmp_lab:$runtime/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_assets:$ISAACLAB_ROOT/source/isaaclab_tasks:$ISAACLAB_ROOT/source/isaaclab_rl:${PYTHONPATH:-}"
exec /root/gpufree-data/yfx/envs/isaaclab/bin/python "$@"
