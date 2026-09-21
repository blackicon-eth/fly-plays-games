"""Turn the ball's horizontal offset into the fly's visual channels, and train a
readout on its descending neurons to pick left or right.

The visual front end is the same `FeatureDetectors` shortcut the Pokémon chapter
uses (the lamina can't relay in a spiking model), so the ball's offset is handed
to the fly's visual projection neurons rather than computed from pixels. The
connectome then runs visual projection neurons -> descending neurons, and the
readout reads a side off that activity. Same one reflex, no memory.
"""
from __future__ import annotations

import math

import numpy as np

from flybrain.eyes import FeatureDetectors
from flybrain.reservoir import Readout, Trace

OBJECT_SIZE = 16.0   # ball angular size handed to the encoder (constant, as in Pokémon Red)
BRAIN_STEPS = 8      # 8 x 20 ms = the 160 ms observation window the readout sees

# A falling object looms as it approaches the paddle. `proximity_size` turns a
# screen y into an angular size so the loom channels (LPLC2/LC4/LPLC1) get a
# bigger drive from an object that is closer. This is what makes urgency explicit:
# a ball about to be lost looms harder than one high up. The mapping is a choice.
PROX_FAR, PROX_NEAR = 50.0, 136.0    # screen y: top of the play field -> the paddle
PROX_MIN, PROX_MAX = 8.0, 32.0       # angular size at those two extremes


def proximity_size(y: float | None, y_far: float = PROX_FAR, y_near: float = PROX_NEAR,
                   smin: float = PROX_MIN, smax: float = PROX_MAX) -> float:
    """Angular size from screen y: closer to the paddle means bigger. Unknown y
    falls back to the constant `OBJECT_SIZE`."""
    if y is None:
        return OBJECT_SIZE
    f = (float(y) - y_far) / (y_near - y_far)
    f = 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)
    return smin + (smax - smin) * f


# Physical urgency. An object at screen y sits `dist = paddle_y - y` from the
# paddle. Its angular size is `size ~ k / dist` (proximity) and its loom rate
# (angular growth per frame) is `loom ~ k * vy / dist**2 = size / t_arrive`
# (imminence). So loom is exactly proximity x imminence, and it is zero for an
# ascending object: a ball low but moving away is not urgent. The paddle also has
# to *reach* the object, so if covering |dx| takes longer than the object takes to
# arrive, loom is boosted. `demand` adds imminence to raw proximity and saturates,
# and `stake` is the cost of missing the object (losing the ball costs a life).
PERSPECTIVE_K = 400.0     # k in size = k / dist
URGENCY_EPS = 6.0         # floor on dist, so a caught ball does not blow up
PADDLE_SPEED = 2.0        # px/frame, to race |dx|
URGENCY_LAMBDA = 20.0     # how much imminence (loom) adds to proximity (size)
URGENCY_TAU = 20.0        # demand -> drive saturation
DRIVE_CAP = 2.0           # most drive a single object can ask for
STAKE_BALL, STAKE_ITEM = 1.0, 0.3   # losing the ball costs a life; an item is a bonus
PADDLE_Y = 136.0


def object_demand(y: float | None, vy: float | None, dx: float | None = None,
                  stake: float = 1.0, paddle_y: float = PADDLE_Y, k: float = PERSPECTIVE_K,
                  eps: float = URGENCY_EPS, paddle_speed: float = PADDLE_SPEED,
                  lam: float = URGENCY_LAMBDA, tau: float = URGENCY_TAU,
                  cap: float = DRIVE_CAP, size_weight: float = 1.0,
                  diagonal: bool = False) -> float:
    """The chase drive an object asks for.

    Proximity (angular size) plus imminence (loom rate, only while falling with
    `vy > 0`), scaled by `stake`, saturating at `cap`. Zero if the object is
    absent, so an ascending or far object asks for little and a falling, close,
    laterally distant one asks for the most.

    `size_weight` scales how much raw proximity counts. Set it low for an object
    whose *position* is already carried by a constant tracking drive (the ball):
    otherwise a close object that is moving away still looks urgent, because its
    size stays large even when its loom rate is zero.

    `diagonal` measures distance as the straight line from the paddle to the object
    (`|dx|` and the vertical gap together) instead of the vertical gap alone. A
    laterally distant object is then *farther* and looms *less*, so it stops asking
    for a detour without a separate reach discount: proximity and imminence both fall
    off smoothly, which a linear readout can use. Only `|dx|` enters the non-diagonal
    loom, as a boost when the paddle cannot cover it in time.
    """
    if y is None:
        return 0.0
    dy = max(float(paddle_y) - float(y), eps)
    v = float(vy or 0.0)
    if diagonal and dx is not None:
        d = math.hypot(float(dx), dy)
        size = k / d
        loom = (k * dy * v / (d * d * d)) if v > 0.0 else 0.0
    else:
        d = dy
        size = k / d
        loom = 0.0
        if v > 0.0:
            loom = k * v / (d * d)
            if dx is not None:
                t_arrive = d / v
                t_cover = abs(float(dx)) / paddle_speed
                loom *= 1.0 + t_cover / t_arrive
    demand = stake * (size_weight * size + lam * loom)
    return cap * demand / (demand + tau)



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
                     grow: float = 0.5, steps: int = BRAIN_STEPS,
                     encoder: dict | None = None) -> np.ndarray:
    """Two objects at once: the ball drives the chase channel (`opp` -> LPLC2/LC4/
    LC10a), a falling item drives the projectile channel (`shots` -> LPLC1). Either
    can be None.

    Both objects' angular size grows by `grow` per step across the window, because
    they are approaching the paddle. This matters: LPLC1 and the loom part of LPLC2
    respond to *growth*, so a frozen object would drive them not at all -- the shot
    channel would be silent and the item could not compete with the ball.

    `encoder` overrides `FeatureDetectors` gains (see `flybrain.eyes.ENCODER`); the
    default front end saturates its drives, which makes the response nearly binary.
    """
    brain.reset(seed=0)
    detectors = FeatureDetectors(brain, **(encoder or {}))
    trace = Trace(brain, types=["descending_neuron"], tau=0.1)
    f = None
    for t in range(steps):
        ramp = 1.0 + grow * t
        opp = None if dx_ball is None else (float(dx_ball), float(size_ball) * ramp)
        shots = () if dx_item is None else (("item", float(dx_item), float(size_item) * ramp),)
        f = trace.observe(brain.step(inject=detectors.inject(opp=opp, shots=shots, threat=0.0)))
    return f


def dn_features_chase2(brain, dx_a: float | None, drive_a: float | None,
                       dx_b: float | None, drive_b: float | None,
                       cap: float = DRIVE_CAP, steps: int = BRAIN_STEPS) -> np.ndarray:
    """Both objects drive the *same* chase channel (LC10a) on their side, with a
    drive set by the caller. Neither pathway is privileged, so the winner is the
    object that asks for more -- feed `object_demand(y, vy, dx, stake)` and the
    more urgent object wins.

    The default `FeatureDetectors.inject` takes only one opponent, so the drive is
    built here directly from `cells["chase"]`; the format matches what `inject`
    returns, so `brain.step` consumes it the same way.
    """
    brain.reset(seed=0)
    det = FeatureDetectors(brain)
    trace = Trace(brain, types=["descending_neuron"], tau=0.1)
    dl = dr = 0.0
    for dx, drive in ((dx_a, drive_a), (dx_b, drive_b)):
        if dx is None or drive is None:
            continue
        d = min(max(float(drive), 0.0), cap)
        if dx < 0:
            dl += d
        else:
            dr += d
    inj = []
    if dl > 0:
        inj.append((det.cells["chase"]["L"], np.float32(dl)))
    if dr > 0:
        inj.append((det.cells["chase"]["R"], np.float32(dr)))
    f = None
    for _ in range(steps):
        f = trace.observe(brain.step(inject=inj))
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
