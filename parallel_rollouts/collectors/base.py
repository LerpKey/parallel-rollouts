"""Collector orchestration shared by all execution backends."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Self

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from ..buffers import PreallocatedRolloutBuffer
from ..policies import PolicyProtocol
from ..seeds import SeedManager
from ..types import PolicyOutput, RolloutBatch

BackendName = Literal["serial", "sync", "async"]
EnvFactory = Callable[[], gym.Env[Any, Any]]


class BackendProtocol:
    """Minimal environment backend interface used by the collector."""

    observation_space: gym.spaces.Box
    action_space: gym.Space[Any]
    num_envs: int

    def reset(self, seed: int | None) -> npt.NDArray[np.float32]:
        raise NotImplementedError

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
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def _normalise_policy_output(
    output: npt.NDArray[Any] | PolicyOutput,
) -> PolicyOutput:
    if isinstance(output, PolicyOutput):
        return output
    return PolicyOutput(actions=np.asarray(output))


def _extract_vector_infos(
    infos: Any,
    num_envs: int,
    observation_shape: tuple[int, ...],
) -> tuple[list[dict[str, Any]], npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Normalize Gymnasium's dict-of-arrays info format."""

    final_observations = np.zeros((num_envs, *observation_shape), dtype=np.float32)
    final_mask = np.zeros(num_envs, dtype=np.bool_)
    normalized = [{} for _ in range(num_envs)]

    if isinstance(infos, (list, tuple)):
        for index, info in enumerate(infos):
            if isinstance(info, dict):
                normalized[index] = dict(info)
                if "final_observation" in info:
                    final_observations[index] = np.asarray(info["final_observation"], dtype=np.float32)
                    final_mask[index] = True
        return normalized, final_observations, final_mask

    if not isinstance(infos, dict):
        return normalized, final_observations, final_mask

    final_values = infos.get("final_observation")
    explicit_mask = infos.get("_final_observation")
    if explicit_mask is None:
        explicit_mask = np.asarray(
            [value is not None for value in final_values] if final_values is not None else [False] * num_envs,
            dtype=np.bool_,
        )
    else:
        explicit_mask = np.asarray(explicit_mask, dtype=np.bool_)

    for index in range(num_envs):
        normalized[index] = {
            key: value[index] if isinstance(value, np.ndarray) and value.shape[:1] == (num_envs,) else value
            for key, value in infos.items()
            if not key.startswith("_")
        }
        if final_values is not None and explicit_mask[index]:
            final_observations[index] = np.asarray(final_values[index], dtype=np.float32)
            final_mask[index] = True
    return normalized, final_observations, final_mask


class ParallelRolloutCollector:
    """Collect fixed-horizon experience from serial or vectorized environments."""

    def __init__(
        self,
        env_fns: Sequence[EnvFactory],
        policy: PolicyProtocol,
        backend: BackendName = "sync",
        horizon: int = 128,
        seed: int | None = 0,
        store_infos: bool = False,
    ) -> None:
        if not env_fns:
            raise ValueError("At least one environment factory is required")
        if backend not in ("serial", "sync", "async"):
            raise ValueError(f"Unknown backend: {backend}")

        from .async_ import AsyncBackend
        from .serial import SerialBackend
        from .sync import SyncBackend

        backend_type = {"serial": SerialBackend, "sync": SyncBackend, "async": AsyncBackend}[backend]
        self.backend_name = backend
        self.policy = policy
        self.seed_manager = SeedManager(seed)
        self._backend: BackendProtocol = backend_type(env_fns)
        self.buffer = PreallocatedRolloutBuffer(
            horizon=horizon,
            num_envs=len(env_fns),
            observation_space=self._backend.observation_space,
            action_space=self._backend.action_space,
            store_infos=store_infos,
        )
        self._observations = self._backend.reset(seed)
        self._closed = False

    @property
    def num_envs(self) -> int:
        return self._backend.num_envs

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._backend.observation_space

    @property
    def action_space(self) -> gym.Space[Any]:
        return self._backend.action_space

    def collect(self) -> RolloutBatch:
        """Collect one horizon and return a view of the preallocated batch."""

        if self._closed:
            raise RuntimeError("Cannot collect after close()")
        for index in range(self.buffer.horizon):
            output = _normalise_policy_output(self.policy(self._observations))
            (
                next_observations,
                rewards,
                terminated,
                truncated,
                infos,
                final_observations,
                final_mask,
            ) = self._backend.step(output.actions)
            self.buffer.write(
                index=index,
                observations=self._observations,
                output=output,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                next_observations=next_observations,
                final_observations=final_observations,
                final_observation_mask=final_mask,
                infos=infos,
            )
            self._observations = next_observations
        return self.buffer.finalize()

    def close(self) -> None:
        if not self._closed:
            self._backend.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
