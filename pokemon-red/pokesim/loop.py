"""Close the loop: the world -> the connectome -> a button -> the world moves.

The behaviour here is "walk toward the nearest map object". A controller turns an
offset into one of the four directions; a rollout presses it and reports, from the
oracle (ground truth, not visible to the controller), how far the nearest object
was after every step. `brain_features` gives the connectome's own view: inject the
offset into the visual channels, let it settle, and read the descending neurons.
"""
from __future__ import annotations

import numpy as np

from flybrain.eyes import FeatureDetectors
from flybrain.reservoir import Trace

from .encoder_b import target as oracle_target

MOVE_FRAMES = 16
SETTLE_FRAMES = 2
DEAD_ZONE = 4.0
BUTTONS = ("down", "right", "up", "left")


def button_toward(dx: float, dy: float, dead_zone: float = DEAD_ZONE) -> str | None:
    """The button that moves toward an object at (dx, dy), or None if it is close."""
    if abs(dx) < dead_zone and abs(dy) < dead_zone:
        return None
    if abs(dx) >= abs(dy):
        return "left" if dx < 0 else "right"
    return "up" if dy < 0 else "down"


def rollout(adapter, controller, steps: int, move_frames: int = MOVE_FRAMES,
            settle: int = SETTLE_FRAMES) -> np.ndarray:
    """Run `controller(adapter) -> button|None` and return the true distance to the
    nearest visible object after each step (NaN when nothing is on screen)."""
    distances = []
    for _ in range(steps):
        button = controller(adapter)
        if button is not None:
            adapter.press(button, move_frames)
            adapter.step(settle)
        t = oracle_target(adapter)
        distances.append(t.distance if t is not None else np.nan)
    return np.array(distances)


def oracle_controller(dead_zone: float = DEAD_ZONE):
    def controller(adapter):
        t = oracle_target(adapter)
        return None if t is None else button_toward(t.dx, t.dy, dead_zone)
    return controller


def random_controller(rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(0)

    def controller(adapter):
        return BUTTONS[rng.integers(len(BUTTONS))]
    return controller


def brain_features(brain, dxs: np.ndarray, steps: int = 8, size: float = 16.0,
                   tau: float = 0.1, seed: int = 0) -> np.ndarray:
    """Descending-neuron trace for each frame, with a fresh brain each time.

    This is the connectome's view of an offset: inject it into the visual
    channels, let it settle for `steps`, read the DNs. A readout on top of this is
    the brain-driven decision.
    """
    detectors = FeatureDetectors(brain)
    out = None
    for i, dx in enumerate(dxs):
        brain.reset(seed=seed)
        detectors.previous = {}
        trace = Trace(brain, types=["descending_neuron"], tau=tau)
        f = None
        for _ in range(steps):
            f = trace.observe(brain.step(inject=detectors.inject(opp=(float(dx), size), threat=0.0)))
        if out is None:
            out = np.zeros((len(dxs), f.shape[0]), np.float32)
        out[i] = f
    return out


def oracle_x_controller(dead_zone: float = 6.0):
    def controller(adapter):
        t = oracle_target(adapter)
        if t is None or abs(t.dx) < dead_zone:
            return None
        return "left" if t.dx < 0 else "right"
    return controller


def rollout_x(adapter, controller, steps: int, move_frames: int = MOVE_FRAMES,
              settle: int = SETTLE_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Like `rollout` but returns the true |dx| and |dy| after each step, so a
    left/right-only controller can be judged on the axis it controls."""
    xs, ys = [], []
    for _ in range(steps):
        button = controller(adapter)
        if button is not None:
            adapter.press(button, move_frames)
            adapter.step(settle)
        t = oracle_target(adapter)
        xs.append(abs(t.dx) if t is not None else np.nan)
        ys.append(abs(t.dy) if t is not None else np.nan)
    return np.array(xs), np.array(ys)


def summarize(distances: np.ndarray, close: float = 20.0) -> dict:
    visible = distances[~np.isnan(distances)]
    return {"visible_frac": float((~np.isnan(distances)).mean()),
            "mean_dist": float(visible.mean()) if len(visible) else float("nan"),
            "min_dist": float(visible.min()) if len(visible) else float("nan"),
            "close_frac": float((visible < close).mean()) if len(visible) else float("nan")}
