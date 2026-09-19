"""Decode the visual side from the connectome's descending-neuron activity.

Inject a known horizontal offset into the connectome's visual channels, keep the
descending-neuron trace per frame, and fit a logistic readout to the true side
with leave-one-block-out CV. A shuffled-label control should sit at chance; the
gap to it is what the connectome actually carries.
"""
from __future__ import annotations

import numpy as np

from flybrain.eyes import FeatureDetectors
from flybrain.reservoir import Readout, Trace


def dn_trace(brain, dxs: np.ndarray, blocks: np.ndarray, size: float = 16.0, threat: float = 0.0,
             tau: float = 0.1, seed: int = 0) -> np.ndarray:
    """Descending-neuron trace while the brain is driven by `dxs`, one per frame."""
    feats = None
    for g in np.unique(blocks):
        idx = np.flatnonzero(blocks == g)
        brain.reset(seed=seed)
        detectors = FeatureDetectors(brain)
        trace = Trace(brain, types=["descending_neuron"], tau=tau)
        for i in idx:
            inject = detectors.inject(opp=(float(dxs[i]), size), threat=threat)
            f = trace.observe(brain.step(inject=inject))
            if feats is None:
                feats = np.zeros((len(dxs), f.shape[0]), np.float32)
            feats[i] = f
    return feats


def decode(brain, dxs: np.ndarray, side: np.ndarray, blocks: np.ndarray) -> Readout:
    feats = dn_trace(brain, dxs, blocks)
    return Readout.fit(feats, side, kind="logistic", groups=blocks)


def compare(brain, dxs: np.ndarray, side: np.ndarray, blocks: np.ndarray) -> dict:
    """The readout on the true offsets, plus a label shuffle as the control."""
    shuffled = np.random.default_rng(0).permutation(side)
    return {"oracle": decode(brain, dxs, side, blocks),
            "shuffled": decode(brain, dxs, shuffled, blocks)}


def _features_per_sample(brain, dxs: np.ndarray, steps: int = 5, size: float = 16.0,
                         threat: float = 0.0, tau: float = 0.1, seed: int = 0) -> np.ndarray:
    """Reset the brain for each frame and drive it `steps` times, so samples are
    independent (no carry-over between frames, which are autocorrelated)."""
    out = None
    for i, dx in enumerate(dxs):
        brain.reset(seed=seed)
        detectors = FeatureDetectors(brain)
        trace = Trace(brain, types=["descending_neuron"], tau=tau)
        f = None
        for _ in range(steps):
            f = trace.observe(brain.step(inject=detectors.inject(opp=(float(dx), size), threat=threat)))
        if out is None:
            out = np.zeros((len(dxs), f.shape[0]), np.float32)
        out[i] = f
    return out


def compare_per_sample(brain, dxs: np.ndarray, side: np.ndarray, blocks: np.ndarray,
                       steps: int = 5) -> dict:
    """Same as `compare` but each frame is an independent, freshly reset trial --
    the protocol of the direct LC10a L/R test. Robust to autocorrelated labels."""
    feats = _features_per_sample(brain, dxs, steps=steps)
    shuffled = np.random.default_rng(0).permutation(side)
    return {"oracle": Readout.fit(feats, side, kind="logistic", groups=blocks),
            "shuffled": Readout.fit(feats, shuffled, kind="logistic", groups=blocks)}
