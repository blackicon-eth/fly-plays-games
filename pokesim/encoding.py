"""Record (frame, oracle target) pairs while the body walks around.

The fly's own visual encoder is trained from what the oracle sees, so the two
must be recorded together: every sample is a grayscale screen plus the nearest
map object's horizontal offset (and side) at that moment. Samples are spaced
`cooldown` moves apart (adjacent frames are near-duplicates and would leak across
a random train/test split) and tagged with the temporal block they came from, so
evaluation can hold whole blocks out.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .encoder_b import target as oracle_target

BUTTONS = ("down", "right", "up", "left")
MOVE_FRAMES = 18
SETTLE_FRAMES = 3


@dataclass
class Dataset:
    """A recorded walk. `frames` (n, H, W) uint8, `side` (n,) 0=L 1=R,
    `dx`/`dy` (n,) true offsets in screen pixels, `blocks` (n,) temporal block ids."""

    frames: np.ndarray
    side: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    blocks: np.ndarray

    def __len__(self) -> int:
        return len(self.frames)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, frames=self.frames, side=self.side, dx=self.dx,
                            dy=self.dy, blocks=self.blocks)

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        d = np.load(path)
        return cls(d["frames"], d["side"], d["dx"], d["dy"], d["blocks"])


def sample_screen(adapter, size: tuple[int, int] = (36, 40)) -> np.ndarray:
    """Grayscale screen downsampled to `size` by block averaging."""
    gray = adapter.screen(gray=True)
    h, w = size
    return gray.reshape(h, gray.shape[0] // h, w, gray.shape[1] // w).mean((1, 3)).astype(np.uint8)


def sweep_policy(rng: np.random.Generator | None = None, phase_moves: int = 200,
                 wiggle: int = 8):
    """Alternate pushing east then west (with a little vertical wiggle) so the
    nearest map object ends up on the left about as often as on the right.

    A plain random walk drifts toward one side of the map, and the nearest NPC
    ends up on the same side almost always -- that is what made the first
    datasets so imbalanced. Alternating the bias balances the classes by design.
    """
    rng = rng or np.random.default_rng(0)

    def policy(step: int) -> str:
        if step % wiggle < wiggle - 2:
            return "right" if (step // phase_moves) % 2 == 0 else "left"
        return "up" if rng.integers(2) else "down"

    return policy


def walk_until(adapter, predicate, max_moves: int = 5000, rng: np.random.Generator | None = None) -> bool:
    """Random-walk until `predicate()` is true. Returns whether it happened."""
    rng = rng or np.random.default_rng(0)
    for _ in range(max_moves):
        adapter.press(BUTTONS[rng.integers(len(BUTTONS))], MOVE_FRAMES)
        adapter.step(SETTLE_FRAMES)
        if predicate():
            return True
    return False


def record(adapter, steps: int, *, cooldown: int = 6, size: tuple[int, int] = (36, 40),
           block_steps: int = 250, policy=None, rng: np.random.Generator | None = None) -> Dataset:
    """Move `steps` times, sampling every `cooldown` moves while an object is visible."""
    rng = rng or np.random.default_rng(0)
    frames, side, dx, dy, blocks = [], [], [], [], []
    for step in range(steps):
        if step % cooldown == 0:
            t = oracle_target(adapter)
            if t is not None:
                frames.append(sample_screen(adapter, size))
                side.append(0 if t.side == "L" else 1)
                dx.append(t.dx)
                dy.append(t.dy)
                blocks.append(step // block_steps)
        button = policy(step) if policy is not None else BUTTONS[rng.integers(len(BUTTONS))]
        adapter.press(button, MOVE_FRAMES)
        adapter.step(SETTLE_FRAMES)
    return Dataset(np.stack(frames), np.array(side, np.int64), np.array(dx, np.float32),
                   np.array(dy, np.float32), np.array(blocks, np.int64))
