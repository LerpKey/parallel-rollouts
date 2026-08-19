from __future__ import annotations

import numpy as np

from examples.vr_edge.synthetic_env import SyntheticVREdgeEnv


def test_synthetic_environment_is_seeded_and_bounded() -> None:
    env_a = SyntheticVREdgeEnv(num_users=3, seed=7)
    env_b = SyntheticVREdgeEnv(num_users=3, seed=7)
    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    assert np.array_equal(obs_a, obs_b)
    next_a = env_a.step(np.ones(3, dtype=np.int64))[0]
    next_b = env_b.step(np.ones(3, dtype=np.int64))[0]
    assert np.array_equal(next_a, next_b)
    assert env_a.observation_space.contains(next_a)
    env_a.close()
    env_b.close()

