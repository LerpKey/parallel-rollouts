"""Preallocated rollout storage."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from .types import PolicyOutput, RolloutBatch


def _action_shape_and_dtype(space: gym.Space[Any]) -> tuple[tuple[int, ...], np.dtype[Any]]:
    if isinstance(space, gym.spaces.Discrete):
        return (), np.dtype(space.dtype)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return tuple(space.shape), np.dtype(space.dtype)
    if isinstance(space, gym.spaces.MultiBinary):
        return tuple(space.shape), np.dtype(space.dtype)
    if isinstance(space, gym.spaces.Box):
        return tuple(space.shape), np.dtype(space.dtype)
    raise TypeError(f"Unsupported action space: {type(space).__name__}")


class PreallocatedRolloutBuffer:
    """Fixed-shape rollout buffer that writes by index instead of appending."""

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        observation_space: gym.spaces.Box,
        action_space: gym.Space[Any],
        store_infos: bool = False,
    ) -> None:
        if horizon <= 0 or num_envs <= 0:
            raise ValueError("horizon and num_envs must be positive")
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("Collector observations must be flattened to a Box space")

        action_shape, action_dtype = _action_shape_and_dtype(action_space)
        self.horizon = horizon
        self.num_envs = num_envs
        self.store_infos = store_infos
        obs_shape = tuple(observation_space.shape)

        self.observations = np.empty((horizon, num_envs, *obs_shape), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((horizon, num_envs, *action_shape), dtype=action_dtype)
        self.rewards = np.empty((horizon, num_envs), dtype=np.float32)
        self.terminated = np.empty((horizon, num_envs), dtype=np.bool_)
        self.truncated = np.empty((horizon, num_envs), dtype=np.bool_)
        self.values = np.zeros((horizon, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((horizon, num_envs), dtype=np.float32)
        self.final_observations = np.zeros_like(self.observations)
        self.final_observation_mask = np.zeros((horizon, num_envs), dtype=np.bool_)
        self.infos: list[list[dict[str, Any]]] | None = [] if store_infos else None

    def reset(self) -> None:
        """Reset write-side state while reusing all allocated arrays."""

        if self.infos is not None:
            self.infos.clear()

    def write(
        self,
        index: int,
        observations: npt.NDArray[np.float32],
        output: PolicyOutput,
        rewards: npt.NDArray[np.float32],
        terminated: npt.NDArray[np.bool_],
        truncated: npt.NDArray[np.bool_],
        next_observations: npt.NDArray[np.float32],
        final_observations: npt.NDArray[np.float32],
        final_observation_mask: npt.NDArray[np.bool_],
        infos: list[dict[str, Any]] | None,
    ) -> None:
        if index < 0 or index >= self.horizon:
            raise IndexError(f"Rollout index {index} is outside [0, {self.horizon})")
        self.observations[index] = observations
        self.actions[index] = output.actions
        self.rewards[index] = rewards
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.next_observations[index] = next_observations
        self.final_observations[index] = final_observations
        self.final_observation_mask[index] = final_observation_mask
        if output.values is not None:
            self.values[index] = output.values
        else:
            self.values[index].fill(0)
        if output.log_probs is not None:
            self.log_probs[index] = output.log_probs
        else:
            self.log_probs[index].fill(0)
        if self.infos is not None:
            self.infos.append(infos or [{} for _ in range(self.num_envs)])

    def finalize(self) -> RolloutBatch:
        """Return a view-backed immutable batch of the allocated storage."""

        return RolloutBatch(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminated=self.terminated,
            truncated=self.truncated,
            next_observations=self.next_observations,
            values=self.values,
            log_probs=self.log_probs,
            final_observations=self.final_observations,
            final_observation_mask=self.final_observation_mask,
            infos=self.infos,
        )
