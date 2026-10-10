#!/usr/bin/env bash
# Run an ultra_eval script in the ultra-sim2sim conda env, pinned to the good CPUs (logical 8/9 are
# defective on this machine), with a clean environment (conda LD_LIBRARY_PATH entries break imports).
#   ultra_eval/run.sh eval_teacher.py --robot ours --effort model --clips 20
#   ultra_eval/run.sh build_ultra_robot.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_BIN="${ULTRA_ENV_BIN:-$HOME/miniconda3/envs/ultra-sim2sim/bin}"
script="$1"; shift
exec env -i HOME="$HOME" PATH="$ENV_BIN:/usr/bin:/bin" MUJOCO_GL=egl PYTHONPATH="${PYTHONPATH:-}" \
     taskset -c 0-7,10-31 "$ENV_BIN/python" "$HERE/$script" "$@"
