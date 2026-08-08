"""A tiny deterministic visual maze for sparse-exploration readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np


class DiscoMazeInfo(TypedDict):
    """Structured state facts emitted with each local-maze observation."""

    position: tuple[int, int]
    goal: tuple[int, int]
    steps: int
    coverage: int


@dataclass(frozen=True)
class DiscreteActionSpace:
    """Minimal discrete action-space API without a Gym dependency."""

    n: int

    def sample(self) -> int:
        """Draw a uniformly random valid action."""
        return int(np.random.randint(self.n))


class RandomDiscoMaze:
    """A local 21x21 maze whose nuisance colours change on every observation.

    The controllable signal is the white agent moving through a fixed maze;
    the randomly coloured floor is deliberately unpredictable.  This makes it
    a small, dependency-free check that an inverse-dynamics embedding resists
    visually salient but action-uncontrollable features.
    """

    size = 21
    action_space = DiscreteActionSpace(4)
    _DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self, max_steps: int = 200, seed: int | None = None) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._walls = self._make_walls()
        self._open_cells = [tuple(cell) for cell in np.argwhere(~self._walls)]
        self.position = (1, 1)
        self.goal = (19, 19)
        self.steps = 0
        self.visited: set[tuple[int, int]] = set()

    @staticmethod
    def _make_walls() -> np.ndarray:
        walls = np.zeros((21, 21), dtype=bool)
        walls[[0, -1], :] = True
        walls[:, [0, -1]] = True
        # Gapped bars give long routes without making any region unreachable.
        for row, gap in ((4, 3), (8, 16), (12, 4), (16, 15)):
            walls[row, 1:20] = True
            walls[row, gap : gap + 2] = False
        for column, gap in ((6, 6), (14, 13)):
            walls[1:20, column] = True
            walls[gap : gap + 3, column] = False
        walls[1, 1] = False
        walls[19, 19] = False
        return walls

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, DiscoMazeInfo]:
        """Reset to a fixed start and a seed-selected reachable goal."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.position = (1, 1)
        candidates = [cell for cell in self._open_cells if cell != self.position]
        self.goal = candidates[int(self._rng.integers(len(candidates)))]
        self.steps = 0
        self.visited = {self.position}
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, DiscoMazeInfo]:
        """Apply an up/right/down/left action and report sparse goal reward."""
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < self.action_space.n:
            raise ValueError("action must be an integer in [0, 3]")
        delta_row, delta_column = self._DELTAS[int(action)]
        candidate = (self.position[0] + delta_row, self.position[1] + delta_column)
        if not self._walls[candidate]:
            self.position = candidate
        self.steps += 1
        self.visited.add(self.position)
        terminated = self.position == self.goal
        truncated = self.steps >= self.max_steps and not terminated
        return self._observation(), float(terminated), terminated, truncated, self._info()

    def _observation(self) -> np.ndarray:
        floor_colour = self._rng.integers(0, 256, size=3, dtype=np.uint8)
        image = np.broadcast_to(floor_colour, (self.size, self.size, 3)).copy()
        image[self._walls] = (0, 0, 0)
        image[self.goal] = (0, 255, 0)
        image[self.position] = (255, 255, 255)
        return image

    def _info(self) -> DiscoMazeInfo:
        return {
            "position": self.position,
            "goal": self.goal,
            "steps": self.steps,
            "coverage": len(self.visited),
        }
