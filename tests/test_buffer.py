from __future__ import annotations

import gymnasium as gym
import numpy as np

from parallel_rollouts.buffers import PreallocatedRolloutBuffer
from parallel_rollouts.types import PolicyOutput


def test_preallocated_buffer_writes_fixed_shapes() -> None:
    buffer = PreallocatedRolloutBuffer(
        horizon=2,
        num_envs=3,
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
        action_space=gym.spaces.Discrete(2),
        store_infos=True,
    )
    observations = np.ones((3, 4), dtype=np.float32)
    buffer.write(
        0,
        observations,
        PolicyOutput(
            actions=np.asarray([0, 1, 0]),
            values=np.asarray([1, 2, 3], dtype=np.float32),
            log_probs=np.asarray([-0.1, -0.2, -0.3], dtype=np.float32),
        ),
        np.asarray([1, 2, 3], dtype=np.float32),
        np.asarray([False, True, False]),
        np.asarray([False, False, False]),
        observations + 1,
        np.zeros_like(observations),
        np.asarray([False, False, False]),
        [{}, {}, {}],
    )
    batch = buffer.finalize()
    assert batch.observations.shape == (2, 3, 4)
    assert batch.actions.shape == (2, 3)
    assert np.array_equal(batch.rewards[0], [1, 2, 3])
    assert np.array_equal(batch.values[0], [1, 2, 3])

