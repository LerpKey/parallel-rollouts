from __future__ import annotations

import numpy as np
import pytest

from examples.vr_edge.synthetic_env import make_env
from parallel_rollouts import ParallelRolloutCollector


def _factories(count: int):
    return [make_env(seed=100 + index, num_users=2, episode_length=8) for index in range(count)]


@pytest.mark.parametrize("backend", ["serial", "sync", "async"])
def test_backend_collects_expected_batch(backend: str) -> None:
    policy = lambda observations: np.ones((observations.shape[0], 2), dtype=np.int64)
    with ParallelRolloutCollector(_factories(2), policy, backend=backend, horizon=5, seed=42) as collector:
        batch = collector.collect()
    assert batch.observations.shape == (5, 2, 10)
    assert batch.actions.shape == (5, 2, 2)
    assert batch.rewards.shape == (5, 2)
    assert batch.dones.shape == (5, 2)


def test_terminal_observation_is_preserved() -> None:
    policy = lambda observations: np.ones((observations.shape[0], 2), dtype=np.int64)
    with ParallelRolloutCollector(_factories(1), policy, backend="serial", horizon=9, seed=42) as collector:
        batch = collector.collect()
    assert batch.final_observation_mask.any()
    assert np.isfinite(batch.final_observations[batch.final_observation_mask]).all()

