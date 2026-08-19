"""Deterministic seed allocation helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedManager:
    """Assign stable, independent seeds to vector-environment workers."""

    base_seed: int | None = 0

    def for_env(self, index: int) -> int | None:
        """Return the seed for one environment."""

        if self.base_seed is None:
            return None
        if index < 0:
            raise ValueError("Environment index must be non-negative")
        return self.base_seed + index

    def all(self, count: int) -> list[int | None]:
        """Return seeds for ``count`` environments."""

        if count < 0:
            raise ValueError("Environment count must be non-negative")
        return [self.for_env(index) for index in range(count)]

