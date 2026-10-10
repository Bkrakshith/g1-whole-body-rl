#!/usr/bin/env bash
# Run a python entry point of g1-whole-body-rl in the g1-mjx conda env with the machine's
# required hygiene (hospital-sim docs/interfaces.md "Machine notes"):
#   * pinned to the good CPUs (logical 8/9 are defective -> random segfaults)
#   * LD_LIBRARY_PATH cleared (a conda env's stale libnvJitLink breaks jax CUDA)
#   * GPU shared with Isaac: XLA may take at most 35 % of the 8 GB 3070 and
#     does not preallocate it
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
exec env -u LD_LIBRARY_PATH taskset -c 0-7,10-31 "$PY" "$@"
