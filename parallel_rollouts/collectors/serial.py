"""Serial backend used as the correctness and performance baseline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ._factory import FlattenedFactory
from .base import EnvFactory, SeedInput


class SerialBackend:
    def __init__(self, env_fns: Sequence[EnvFactory]) -> None:
        self.envs = [FlattenedFactory(factory)() for factory in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        for env in self.envs[1:]:
            if env.observation_space != self.observation_space or env.action_space != self.action_space:
                self.close()
                raise ValueError("All environments must have matching spaces")

    def reset(self, seed: SeedInput) -> npt.NDArray[np.float32]:
        observations = []
        for index, env in enumerate(self.envs):
            env_seed = seed[index] if isinstance(seed, list) else seed
            observation, _ = env.reset(seed=env_seed)
            observations.append(observation)
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
        observations: list[npt.NDArray[np.float32]] = []
        rewards = np.empty(self.num_envs, dtype=np.float32)
        terminated = np.empty(self.num_envs, dtype=np.bool_)
        truncated = np.empty(self.num_envs, dtype=np.bool_)
        infos: list[dict[str, Any]] = []
        final_observations = np.zeros((self.num_envs, *self.observation_space.shape), dtype=np.float32)
        final_mask = np.zeros(self.num_envs, dtype=np.bool_)

        for index, env in enumerate(self.envs):
            observation, reward, term, trunc, info = env.step(actions[index])
            done = bool(term or trunc)
            if done:
                final_observations[index] = observation
                final_mask[index] = True
                observation, reset_info = env.reset()
                info = {**reset_info, **info, "final_observation": final_observations[index].copy()}
            observations.append(np.asarray(observation, dtype=np.float32))
            rewards[index] = reward
            terminated[index] = term
            truncated[index] = trunc
            infos.append(dict(info))

        return (
            np.asarray(observations, dtype=np.float32),
            rewards,
            terminated,
            truncated,
            infos,
            final_observations,
            final_mask,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()
