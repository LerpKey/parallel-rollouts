"""Optional CartPole smoke example using Gymnasium's classic-control extra."""

from __future__ import annotations

import argparse
import multiprocessing as mp

import gymnasium as gym

from parallel_rollouts import ParallelRolloutCollector, RandomPolicy


def make_cartpole() -> gym.Env:
    return gym.make("CartPole-v1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["serial", "sync", "async"], default="sync")
    parser.add_argument("--num-envs", type=int, default=4)
    args = parser.parse_args()
    env_fns = [make_cartpole for _ in range(args.num_envs)]
    with ParallelRolloutCollector(
        env_fns=env_fns,
        policy=RandomPolicy(gym.spaces.Discrete(2), seed=42),
        backend=args.backend,
        horizon=128,
    ) as collector:
        batch = collector.collect()
        print(f"Collected {batch.rewards.size} transitions with {args.backend} backend")


if __name__ == "__main__":
    mp.freeze_support()
    main()

