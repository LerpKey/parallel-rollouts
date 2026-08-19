"""Policy protocols and small adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from .types import PolicyOutput


class PolicyProtocol(Protocol):
    """A policy callable that accepts a batch of flattened observations."""

    def __call__(
        self, observations: npt.NDArray[np.float32]
    ) -> npt.NDArray[Any] | PolicyOutput:
        ...


class SingleObservationPolicy:
    """Adapt a single-observation callable to the batched policy protocol."""

    def __init__(self, policy: Callable[[npt.NDArray[np.float32]], Any]) -> None:
        self.policy = policy

    def __call__(self, observations: npt.NDArray[np.float32]) -> npt.NDArray[Any]:
        actions = [self.policy(observation) for observation in observations]
        return np.asarray(actions)


class RandomPolicy:
    """Deterministic-by-seed random policy useful for smoke tests and benchmarks."""

    def __init__(self, action_space: gym.Space[Any], seed: int | None = 0) -> None:
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def __call__(self, observations: npt.NDArray[np.float32]) -> npt.NDArray[Any]:
        batch_size = observations.shape[0]
        if isinstance(self.action_space, gym.spaces.Discrete):
            return self.rng.integers(0, self.action_space.n, size=batch_size, dtype=self.action_space.dtype)
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return np.stack(
                [self.rng.integers(0, high, size=batch_size, dtype=np.int64) for high in self.action_space.nvec],
                axis=1,
            )
        if isinstance(self.action_space, gym.spaces.Box):
            return self.rng.uniform(
                self.action_space.low,
                self.action_space.high,
                size=(batch_size, *self.action_space.shape),
            ).astype(self.action_space.dtype)
        raise TypeError(f"RandomPolicy does not support {type(self.action_space).__name__}")
