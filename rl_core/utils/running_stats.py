"""Numerically stable streaming statistics used by intrinsic rewards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunningMeanStd:
    """Track a population mean and variance without retaining samples.

    The implementation uses the parallel variance update, so batching samples
    produces the same result as adding them one at a time (up to round-off).
    """

    shape: tuple[int, ...] = ()
    epsilon: float = 1e-4

    def __post_init__(self) -> None:
        """Initialize the empty population estimate."""
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(self.epsilon)

    def update(self, values: np.ndarray | float) -> None:
        """Incorporate a batch whose trailing dimensions equal ``shape``."""
        array = np.asarray(values, dtype=np.float64)
        if self.shape:
            if array.ndim < len(self.shape) or array.shape[-len(self.shape) :] != self.shape:
                raise ValueError(f"expected trailing shape {self.shape}, got {array.shape}")
            batch = array.reshape((-1, *self.shape))
        else:
            batch = array.reshape(-1)
        if batch.shape[0] == 0:
            return
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = float(batch.shape[0])
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + np.square(delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        """Return a non-negative population standard deviation."""
        return np.sqrt(self.var)

    def normalize(self, values: np.ndarray | float, clip: float | None = None) -> np.ndarray:
        """Standardize values with the current statistics without updating them."""
        result = (np.asarray(values, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        if clip is not None:
            if clip <= 0:
                raise ValueError("clip must be positive")
            result = np.clip(result, -clip, clip)
        return result
