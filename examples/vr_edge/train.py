"""Run a small PPO training smoke test on the synthetic VR environment."""

from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np
import torch

from parallel_rollouts import ParallelRolloutCollector

from .policy import ActorCriticPolicy, PPOTrainer
from .synthetic_env import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["serial", "sync", "async"], default="sync")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    env_fns = [make_env(seed=42 + index) for index in range(args.num_envs)]
    policy = ActorCriticPolicy(observation_dim=8 * 5, num_users=8)
    trainer = PPOTrainer(policy)
    with ParallelRolloutCollector(
        env_fns=env_fns,
        policy=policy,
        backend=args.backend,
        horizon=args.horizon,
        seed=42,
    ) as collector:
        for update in range(args.updates):
            metrics = trainer.update(collector.collect())
            print(f"update={update + 1} loss={metrics['loss']:.4f} reward={metrics['mean_reward']:.4f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

