#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
if [[ -n "$mode" ]]; then shift; fi
case "$mode" in
  -t|--train) script=train.py; args=(--headless) ;;
  -p|--play) script=play.py; args=() ;;
  -v|--terrain-view) script=view_terrain.py; args=() ;;
  -e|--export) exec conda run -n env_isaaclab python "$root/scripts/export.py" "$@" ;;
  *) echo 'Usage: bash/leggedskill.sh {-t|-p|-v|-e} --robot {go1|go2} [options]'; exit 2 ;;
esac
exec conda run --no-capture-output -n env_isaaclab env TERM=xterm-256color "${ISAACLAB_ROOT:-/home/re/IsaacLab}/isaaclab.sh" -p "$root/scripts/$script" "${args[@]}" "$@"
