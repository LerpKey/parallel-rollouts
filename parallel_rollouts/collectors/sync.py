"""Same-process Gymnasium vector backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from ._factory import FlattenedFactory
from .base import EnvFactory, _extract_vector_infos


class SyncBackend:
    def __init__(self, env_fns: Sequence[EnvFactory]) -> None:
        factories = [FlattenedFactory(factory) for factory in env_fns]
        self.envs = gym.vector.SyncVectorEnv(factories)
        self.num_envs = len(factories)
        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space

    def reset(self, seed: int | None) -> npt.NDArray[np.float32]:
        observations, _ = self.envs.reset(seed=seed)
        return np.asarray(observations, dtype=np.float32)

    def step(
        self, actions: npt.NDArray[Any]
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
        list[dict[str, Any]],
        npt.NDArray[np.float32],
        npt.NDArray[np.bool_],
    ]:
        observations, rewards, terminated, truncated, infos = self.envs.step(actions)
        normalized, final_obs, final_mask = _extract_vector_infos(
            infos, self.num_envs, self.observation_space.shape
        )
        return (
            np.asarray(observations, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
            normalized,
            final_obs,
            final_mask,
        )

    def close(self) -> None:
        self.envs.close()
