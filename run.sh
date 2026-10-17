#!/usr/bin/env bash
# Run a python entry point of g1-whole-body-rl in the g1-mjx conda env.
#   * LD_LIBRARY_PATH cleared (a conda env's stale libnvJitLink breaks jax CUDA)
#   * XLA takes at most 35 % of the GPU by default and does not preallocate it, so the
#     GPU can be shared with Isaac Sim; override with XLA_PYTHON_CLIENT_MEM_FRACTION
#   * optional CPU pinning: set G1_MJX_CPUS to a taskset list (for example "0-7,10-31")
#     to keep training off cores you want to reserve; unset by default
# Usage: ./run.sh scripts/bench.py --backend gpu      (paths relative to this folder)
#        ./run.sh -m pytest tests -q
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${G1_MJX_PYTHON:-$HOME/miniconda3/envs/g1-mjx/bin/python}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# XLA autotuner/triton fallbacks log whole HLO fusions at ERROR level: quiet
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$HERE"
if [ -n "${G1_MJX_CPUS:-}" ]; then
  exec env -u LD_LIBRARY_PATH taskset -c "$G1_MJX_CPUS" "$PY" "$@"
fi
exec env -u LD_LIBRARY_PATH "$PY" "$@"
