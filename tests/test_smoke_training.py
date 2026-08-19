from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch extra is not installed")
def test_minimal_ppo_smoke() -> None:
    import torch

    from examples.vr_edge.policy import ActorCriticPolicy, PPOTrainer
    from examples.vr_edge.synthetic_env import make_env
    from parallel_rollouts import ParallelRolloutCollector

    torch.manual_seed(0)
    policy = ActorCriticPolicy(observation_dim=10, num_users=2)
    trainer = PPOTrainer(policy)
    with ParallelRolloutCollector(
        [make_env(1, num_users=2), make_env(2, num_users=2)],
        policy,
        backend="sync",
        horizon=16,
        seed=0,
    ) as collector:
        metrics = trainer.update(collector.collect(), epochs=1, minibatch_size=8)
    assert metrics["loss"] == metrics["loss"]

