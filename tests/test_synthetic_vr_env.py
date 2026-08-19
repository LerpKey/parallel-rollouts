from __future__ import annotations

import json
from pathlib import Path

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


def test_public_fixture_has_documented_schema() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "synthetic_trace.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "synthetic-bandwidth-v1"
    assert len(payload["bandwidth_mbps"]) == 2
    assert all(len(row) == 8 for row in payload["bandwidth_mbps"])
