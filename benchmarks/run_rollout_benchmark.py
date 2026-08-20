"""Measure rollout throughput on native Windows, CPU first."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import sys
import time
from functools import partial
from pathlib import Path

import gymnasium
import numpy as np

from examples.vr_edge.synthetic_env import SyntheticVREdgeEnv
from parallel_rollouts import ParallelRolloutCollector


class ConstantPolicy:
    def __init__(self, num_users: int, action: int = 1) -> None:
        self.actions = np.full(num_users, action, dtype=np.int64)

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return np.repeat(self.actions[None, :], observations.shape[0], axis=0)


def measure(
    backend: str,
    num_envs: int,
    num_users: int,
    warmup: int,
    steps: int,
    repeats: int,
) -> dict:
    timings = []
    horizon = 256
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if steps <= 0 or steps % horizon != 0:
        raise ValueError(f"steps must be a positive multiple of horizon ({horizon})")
    warmup_steps_actual = ((warmup + horizon - 1) // horizon) * horizon
    for repeat in range(repeats):
        env_fns = [
            partial(SyntheticVREdgeEnv, num_users=num_users, episode_length=256, seed=1000 + i)
            for i in range(num_envs)
        ]
        policy = ConstantPolicy(num_users=num_users)
        with ParallelRolloutCollector(env_fns, policy, backend=backend, horizon=horizon, seed=42) as collector:
            remaining_warmup = warmup_steps_actual
            while remaining_warmup > 0:
                collector.collect()
                remaining_warmup -= horizon
            start = time.perf_counter()
            remaining = steps
            while remaining > 0:
                collector.collect()
                remaining -= horizon
            timings.append(time.perf_counter() - start)
    elapsed = float(np.median(timings))
    transitions = num_envs * steps
    return {
        "backend": backend,
        "num_envs": num_envs,
        "num_users": num_users,
        "horizon": horizon,
        "warmup_steps_requested": warmup,
        "warmup_steps_actual": warmup_steps_actual,
        "steps_per_env": steps,
        "transitions_per_second": transitions / elapsed,
        "elapsed_seconds_median": elapsed,
        "elapsed_seconds_repeats": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["serial", "sync", "async"], default="sync")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-users", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(
        args.backend,
        args.num_envs,
        args.num_users,
        args.warmup_steps,
        args.steps,
        args.repeats,
    )
    result["python"] = sys.version
    result["numpy"] = np.__version__
    result["gymnasium"] = gymnasium.__version__
    result["platform"] = platform.platform()
    result["processor"] = platform.processor()
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    mp.freeze_support()
    main()
