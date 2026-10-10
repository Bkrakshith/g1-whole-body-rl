"""Run one of ULTRA's own MuJoCo sim2sim scripts unchanged, on a CPU-only torch.

ULTRA's motion clips and checkpoint were saved from CUDA tensors; this wrapper maps every torch.load to the CPU
and then executes the upstream script with its own arguments.

    python ../../ultra_eval/run_upstream.py ultra/sim2sim_teacher.py --ckpt ... --motion_path ... --record_video
(run from third_party/ULTRA, with PYTHONPATH=ultra; see ultra_eval/run.sh)
"""
import functools
import runpy
import sys

import torch

torch.load = functools.partial(torch.load, map_location="cpu", weights_only=False)

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
