#!/usr/bin/env bash
# Run an ultra_eval script in the ultra-sim2sim conda env with a clean environment (conda
# LD_LIBRARY_PATH entries break imports). Set G1_MJX_CPUS to a taskset list to pin CPUs.
#   ultra_eval/run.sh eval_teacher.py --robot ours --effort model --clips 20
#   ultra_eval/run.sh build_ultra_robot.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_BIN="${ULTRA_ENV_BIN:-$HOME/miniconda3/envs/ultra-sim2sim/bin}"
script="$1"; shift
PIN=()
[ -n "${G1_MJX_CPUS:-}" ] && PIN=(taskset -c "$G1_MJX_CPUS")
exec env -i HOME="$HOME" PATH="$ENV_BIN:/usr/bin:/bin" MUJOCO_GL=egl PYTHONPATH="${PYTHONPATH:-}" \
     ${PIN[@]+"${PIN[@]}"} "$ENV_BIN/python" "$HERE/$script" "$@"
