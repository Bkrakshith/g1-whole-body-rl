#!/usr/bin/env python
"""Throughput benchmark of the REAL training env step (PD substeps, per-env
model fields, rewards, contacts, auto-reset wrapper) -> choose ppo.num_envs
(skill Phase 1). Hardest stage (full DR, pushes, manip) = honest cost.

  ./run.sh scripts/bench.py --backend gpu --num_envs 256,1024,2048,4096,8192
Results -> outputs/bench.json. GPU memory is shared (XLA <= 35 %, no
preallocation, see run.sh); a size that OOMs is reported and skipped.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from g1_wbc.backend import select_backend  # noqa: E402


def bench(sizes, steps, backend, curriculum):
    import jax
    import jax.numpy as jnp
    from g1_wbc.config import RL_ROOT, load_config
    from g1_wbc.env import G1TeleopEnv
    from g1_wbc.wrappers import wrap_for_training
    cfg = load_config()
    env = G1TeleopEnv(cfg, cfg.stages(curriculum)[-1])
    res = {}
    for n in sizes:
        try:
            w = wrap_for_training(env, episode_length=cfg.train.ppo.episode_length)
            reset, step = jax.jit(w.reset), jax.jit(w.step)
            s = reset(jax.random.split(jax.random.PRNGKey(0), n))
            a = jnp.zeros((n, env.action_size))
            t_c = time.perf_counter()
            s = step(s, a)
            jax.block_until_ready(s.obs["state"])
            t_compile = time.perf_counter() - t_c
            t0 = time.perf_counter()
            for _ in range(steps):
                s = step(s, a)
            jax.block_until_ready(s.obs["state"])
            dt = time.perf_counter() - t0
            nan = bool(jnp.any(jnp.isnan(s.obs["state"])))
            res[n] = {"env_steps_per_s": n * steps / dt, "ms_per_step": 1e3 * dt / steps,
                      "compile_s": t_compile, "nan": nan}
            print(f"num_envs={n:6d} {n * steps / dt:10,.0f} env steps/s  "
                  f"{1e3 * dt / steps:7.1f} ms/step  compile {t_compile:5.1f}s  nan={nan}", flush=True)
        except Exception as e:  # OOM on the shared GPU
            print(f"num_envs={n:6d} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            res[n] = {"error": str(e)[:300]}
            break
    out = os.path.join(RL_ROOT, "outputs", "bench.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allr = json.load(open(out)) if os.path.exists(out) else {}
    allr[backend] = {str(k): v for k, v in res.items()}
    json.dump(allr, open(out, "w"), indent=2)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", default="256,1024,2048,4096")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--curriculum", default="teleop")
    a = ap.parse_args()
    b = select_backend(a.backend)
    bench([int(x) for x in a.num_envs.split(",")], a.steps, b, a.curriculum)


if __name__ == "__main__":
    main()
