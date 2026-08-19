from __future__ import annotations

import numpy as np

from examples.vr_edge.synthetic_env import make_env
from parallel_rollouts import ParallelRolloutCollector


def _collect(seed: int) -> tuple[np.ndarray, np.ndarray]:
    policy = lambda observations: np.ones((observations.shape[0], 2), dtype=np.int64)
    with ParallelRolloutCollector(
        [make_env(seed=10 + i, num_users=2) for i in range(2)],
        policy,
        backend="sync",
        horizon=12,
        seed=seed,
    ) as collector:
        batch = collector.collect()
    return batch.observations.copy(), batch.rewards.copy()


def test_same_seed_reproduces_rollout() -> None:
    observations_a, rewards_a = _collect(42)
    observations_b, rewards_b = _collect(42)
    assert np.array_equal(observations_a, observations_b)
    assert np.array_equal(rewards_a, rewards_b)


def test_different_seed_changes_rollout() -> None:
    observations_a, _ = _collect(42)
    observations_b, _ = _collect(43)
    assert not np.array_equal(observations_a, observations_b)


def test_multiple_collections_advance_without_reallocating_info_list() -> None:
    policy = lambda observations: np.ones((observations.shape[0], 2), dtype=np.int64)
    with ParallelRolloutCollector(
        [make_env(seed=10 + i, num_users=2) for i in range(2)],
        policy,
        backend="sync",
        horizon=4,
        seed=42,
        store_infos=True,
    ) as collector:
        first = collector.collect()
        second = collector.collect()
    assert first.infos is second.infos
    assert len(second.infos or []) == 4
