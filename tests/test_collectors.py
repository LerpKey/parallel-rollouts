from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from examples.vr_edge.synthetic_env import make_env
from parallel_rollouts import ParallelRolloutCollector


class DictObservationEnv(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "position": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
            "status": gym.spaces.Discrete(3),
        }
    )
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return {"position": np.zeros(2, dtype=np.float32), "status": 0}, {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("invalid action")
        self.step_count += 1
        return (
            {"position": np.full(2, self.step_count, dtype=np.float32), "status": 1},
            1.0,
            False,
            self.step_count >= 4,
            {},
        )


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


def test_dict_observation_is_flattened_at_collector_boundary() -> None:
    policy = lambda observations: np.zeros(observations.shape[0], dtype=np.int64)
    with ParallelRolloutCollector(
        [DictObservationEnv, DictObservationEnv],
        policy,
        backend="sync",
        horizon=2,
        seed=42,
    ) as collector:
        batch = collector.collect()
    assert isinstance(collector.observation_space, gym.spaces.Box)
    assert batch.observations.shape == (2, 2, 5)
