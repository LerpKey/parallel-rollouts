"""Small, deterministic VR-edge workload with no external dataset."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt


class SyntheticVREdgeEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.int64]]):
    """Multi-user bitrate control with synthetic bandwidth fluctuations.

    The environment intentionally models only the application shape needed by
    the example: several users share a fluctuating edge link, each user picks
    a bitrate, and the reward balances quality, stalls, and smoothness.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}
    BITRATES: ClassVar[npt.NDArray[np.float32]] = np.asarray([2.0, 4.0, 8.0, 12.0], dtype=np.float32)

    def __init__(
        self,
        num_users: int = 8,
        episode_length: int = 256,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if num_users <= 0:
            raise ValueError("num_users must be positive")
        self.num_users = num_users
        self.episode_length = episode_length
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_users, 5),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete(
            np.full(num_users, len(self.BITRATES), dtype=np.int64)
        )
        self._seed = seed
        self.current_step = 0
        self.bandwidth = np.empty(num_users, dtype=np.float32)
        self.buffer = np.empty(num_users, dtype=np.float32)
        self.current_bitrate = np.empty(num_users, dtype=np.int64)

    def _observation(self) -> npt.NDArray[np.float32]:
        obs = np.empty((self.num_users, 5), dtype=np.float32)
        obs[:, 0] = self.bandwidth / 20.0
        obs[:, 1] = self.buffer / 5.0
        obs[:, 2] = self.current_bitrate / (len(self.BITRATES) - 1)
        obs[:, 3] = np.arange(self.num_users, dtype=np.float32) / max(self.num_users - 1, 1)
        obs[:, 4] = self.current_step / max(self.episode_length, 1)
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        del options
        super().reset(seed=seed if seed is not None else self._seed)
        self.current_step = 0
        self.bandwidth.fill(12.0)
        self.bandwidth += self.np_random.uniform(-2.0, 2.0, size=self.num_users).astype(np.float32)
        self.buffer.fill(2.5)
        self.current_bitrate.fill(1)
        return self._observation(), {"seed": seed if seed is not None else self._seed}

    def step(
        self, action: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.num_users,) or not self.action_space.contains(action):
            raise ValueError(f"Expected action shape {(self.num_users,)}, got {action.shape}")

        bitrate = self.BITRATES[action]
        previous = self.BITRATES[self.current_bitrate]
        effective_share = self.bandwidth / self.num_users
        delivered = effective_share / np.maximum(bitrate, 1e-6)
        self.buffer = np.clip(self.buffer + delivered - 1.0, 0.0, 5.0)
        stalls = self.buffer <= 1e-6
        quality = bitrate / self.BITRATES[-1]
        smoothness = np.abs(bitrate - previous) / self.BITRATES[-1]
        reward = float(np.mean(quality - 1.5 * stalls.astype(np.float32) - 0.05 * smoothness))

        noise = self.np_random.normal(0.0, 1.5, size=self.num_users).astype(np.float32)
        self.bandwidth = np.clip(0.92 * self.bandwidth + 0.96 + noise, 2.0, 40.0)
        self.current_bitrate[:] = action
        self.current_step += 1
        truncated = self.current_step >= self.episode_length
        info = {
            "mean_quality": float(np.mean(quality)),
            "stall_rate": float(np.mean(stalls)),
            "mean_bandwidth": float(np.mean(self.bandwidth)),
        }
        return self._observation(), reward, False, truncated, info


def make_env(
    seed: int,
    num_users: int = 8,
    episode_length: int = 256,
) -> Callable[[], SyntheticVREdgeEnv]:
    """Return a pickle-friendly environment factory for Windows spawn."""

    return partial(
        SyntheticVREdgeEnv,
        num_users=num_users,
        episode_length=episode_length,
        seed=seed,
    )
