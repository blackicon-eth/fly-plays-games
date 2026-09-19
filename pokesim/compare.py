"""Compare encoder A (pixels) and encoder B (oracle) as drivers of the connectome.

Both feed the same `FeatureDetectors` channels; only the horizontal offset differs
(A's is predicted from pixels). Each temporal block is run on a freshly reset
brain, so blocks stay independent, and the descending-neuron trace is kept per
frame. A logistic readout then decodes the true side from that trace with
leave-one-block-out CV. The gap between B and A is what the visual bottleneck
costs; a shuffled-label control should sit at chance.
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


def compare(brain, dataset, dx_a: np.ndarray) -> dict:
    """Readouts for the oracle (B) and the visual encoder (A), plus a label shuffle."""
    out = {"B_oracle": decode(brain, dataset.dx, dataset.side, dataset.blocks),
           "A_vision": decode(brain, dx_a, dataset.side, dataset.blocks)}
    shuffled = np.random.default_rng(0).permutation(dataset.side)
    out["shuffled"] = decode(brain, dataset.dx, shuffled, dataset.blocks)
    return out


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


def compare_per_sample(brain, dataset, dx_a: np.ndarray, steps: int = 5) -> dict:
    """Same as `compare` but each frame is an independent, freshly reset trial --
    the protocol of the direct LC10a L/R test. Robust to autocorrelated labels."""
    out = {}
    for name, dxs in [("B_oracle", dataset.dx), ("A_vision", dx_a)]:
        feats = _features_per_sample(brain, dxs, steps=steps)
        out[name] = Readout.fit(feats, dataset.side, kind="logistic", groups=dataset.blocks)
    shuffled = np.random.default_rng(0).permutation(dataset.side)
    feats = _features_per_sample(brain, dataset.dx, steps=steps)
    out["shuffled"] = Readout.fit(feats, shuffled, kind="logistic", groups=dataset.blocks)
    return out
