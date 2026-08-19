"""Pickle-friendly environment factory wrapper for Windows spawn."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym


class FlattenedFactory:
    """Wrap a top-level/pickleable factory with Gymnasium's flatten wrapper."""

    def __init__(self, factory: Callable[[], gym.Env[Any, Any]]) -> None:
        self.factory = factory

    def __call__(self) -> gym.Env[Any, Any]:
        return gym.wrappers.FlattenObservation(self.factory())
