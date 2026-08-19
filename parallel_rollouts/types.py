"""Public data types used by the rollout runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PolicyOutput:
    """Output returned by a batched policy.

    ``actions`` is required. ``log_probs`` and ``values`` are optional and are
    stored by the collector when supplied, which makes the collector usable
    with PPO-style learners without coupling the runtime to one algorithm.
    """

    actions: npt.NDArray[Any]
    log_probs: npt.NDArray[np.float32] | None = None
    values: npt.NDArray[np.float32] | None = None


@dataclass(frozen=True)
class RolloutBatch:
    """A fixed-horizon batch with leading dimensions ``[time, env, ...]``."""

    observations: npt.NDArray[np.float32]
    actions: npt.NDArray[Any]
    rewards: npt.NDArray[np.float32]
    terminated: npt.NDArray[np.bool_]
    truncated: npt.NDArray[np.bool_]
    next_observations: npt.NDArray[np.float32]
    values: npt.NDArray[np.float32]
    log_probs: npt.NDArray[np.float32]
    final_observations: npt.NDArray[np.float32]
    final_observation_mask: npt.NDArray[np.bool_]
    infos: list[list[dict[str, Any]]] | None = None

    @property
    def dones(self) -> npt.NDArray[np.bool_]:
        """Return the union of terminated and truncated flags."""

        return np.logical_or(self.terminated, self.truncated)

