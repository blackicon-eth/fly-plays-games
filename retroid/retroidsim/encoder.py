"""Turn the ball's horizontal offset into the fly's visual channels, and train a
readout on its descending neurons to pick left or right.

The visual front end is the same `FeatureDetectors` shortcut the Pokémon chapter
uses (the lamina can't relay in a spiking model), so the ball's offset is handed
to the fly's visual projection neurons rather than computed from pixels. The
connectome then runs visual projection neurons -> descending neurons, and the
readout reads a side off that activity. Same one reflex, no memory.
"""
from __future__ import annotations

import numpy as np

from flybrain.eyes import FeatureDetectors
from flybrain.reservoir import Readout, Trace

OBJECT_SIZE = 16.0   # ball angular size handed to the encoder (constant, as in Pokémon Red)
BRAIN_STEPS = 8      # 8 x 20 ms = the 160 ms observation window the readout sees


def dn_features(brain, dx: float, size: float = OBJECT_SIZE, steps: int = BRAIN_STEPS) -> np.ndarray:
    """Run one independent decision: reset the brain, inject the ball's offset,
    and return the descending-neuron trace after `steps` steps."""
    brain.reset(seed=0)
    detectors = FeatureDetectors(brain)
    trace = Trace(brain, types=["descending_neuron"], tau=0.1)
    f = None
    for _ in range(steps):
        f = trace.observe(brain.step(inject=detectors.inject(opp=(float(dx), float(size)), threat=0.0)))
    return f


def dn_features_pair(brain, dx_ball: float | None, dx_item: float | None,
                     size_ball: float = OBJECT_SIZE, size_item: float = OBJECT_SIZE,
                     grow: float = 0.5, steps: int = BRAIN_STEPS) -> np.ndarray:
    """Two objects at once: the ball drives the chase channel (`opp` -> LPLC2/LC4/
    LC10a), a falling item drives the projectile channel (`shots` -> LPLC1). Either
    can be None.

    Both objects' angular size grows by `grow` per step across the window, because
    they are approaching the paddle. This matters: LPLC1 and the loom part of LPLC2
    respond to *growth*, so a frozen object would drive them not at all -- the shot
    channel would be silent and the item could not compete with the ball.
    """
    brain.reset(seed=0)
    detectors = FeatureDetectors(brain)
    trace = Trace(brain, types=["descending_neuron"], tau=0.1)
    f = None
    for t in range(steps):
        ramp = 1.0 + grow * t
        opp = None if dx_ball is None else (float(dx_ball), float(size_ball) * ramp)
        shots = () if dx_item is None else (("item", float(dx_item), float(size_item) * ramp),)
        f = trace.observe(brain.step(inject=detectors.inject(opp=opp, shots=shots, threat=0.0)))
    return f


def train_readout(brain, span: float = 80.0, samples: int = 33, verbose: bool = False) -> Readout:
    """Fit a left/right readout on a sweep of ball offsets, labelled by side.

    The label is just `dx > 0`, so the readout learns the reflex "ball on my
    right, go right". There is no reward and no learning in the connectome; the
    only trained weights are this linear map over the descending neurons.
    """
    dxs = np.linspace(-span, span, samples)
    feats = np.stack([dn_features(brain, dx) for dx in dxs])
    return Readout.fit(feats, (dxs > 0).astype(int), kind="logistic", verbose=verbose)
