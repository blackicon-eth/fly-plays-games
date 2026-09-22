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
# ascending object: a ball low but moving away is not urgent. `demand` adds
# imminence to raw proximity and saturates. There is no reach term: whether the
# paddle can cover |dx| in time is a fact about the actor, not the stimulus, so it
# is left out. `stake` is the cost of missing the object (losing the ball costs a life).
PERSPECTIVE_K = 400.0     # k in size = k / dist
URGENCY_EPS = 6.0         # floor on dist, so a caught ball does not blow up
URGENCY_LAMBDA = 20.0     # how much imminence (loom) adds to proximity (size)
URGENCY_TAU = 20.0        # demand -> drive saturation
DRIVE_CAP = 2.0           # most drive a single object can ask for
STAKE_BALL, STAKE_ITEM = 1.0, 0.3   # losing the ball costs a life; an item is a bonus
PADDLE_Y = 136.0
VX_GAIN = 0.4             # drive added on the side an object is moving toward, per px/frame of `vx`
STAKE_ESCAPE = 1.0        # how much the boss's shot counts as a threat to flee
ESCAPE_CAP = 3.0          # most escape drive one shot can ask for
# A shot has to loom enough to move the paddle clear before it reaches the edge.
# The ball's own drive leans on imminence (size weight 0.1); a shot is a smaller
# object on a shorter fuse, so it is read with more of its raw proximity.
ESCAPE_SIZE_WEIGHT = 0.4
# The escape neurons are sensitive (their tonic alone parks them near threshold),
# so any drive at all would make them fire. Only hand them a shot that is actually
# bearing down on the paddle; below this the shot is somewhere else on the screen.
ESCAPE_MIN = 1.0


def escape_sides(projectiles, paddle_x: float, prev=(), stake: float = STAKE_ESCAPE,
                 cap: float = ESCAPE_CAP, size_weight: float = ESCAPE_SIZE_WEIGHT,
                 min_drive: float = ESCAPE_MIN) -> tuple[float, float]:
    """Per-side escape drive from the boss's shots: the looming urgency of a shot
    that is about to reach the paddle, delivered to the escape neuron (DNp01) on
    the shot's own side. The fly's escape reflex turns it *away* from that side;
    the readout reads that reflex off the descending trace.

    `prev` is last frame's shot list, used only for the present descent speed (vy,
    a visible motion, no extrapolation). `size_weight` keeps the drive about
    imminence, not raw proximity, so it stays low while the shot is high up and
    rises only as it bears down on the paddle -- otherwise the fly would flinch at
    every shot and never track the ball.
    """
    dl = dr = 0.0
    for x, y in projectiles:
        dx = float(x) - float(paddle_x)
        vy, nearest = 0.0, None
        for px, py in prev:
            d = abs(px - x)
            if nearest is None or d < nearest:
                nearest, vy = d, float(y) - float(py)
        if nearest is None or nearest > 16:
            vy = 0.0
        d = object_demand(y, max(vy, 0.0), dx, stake=stake, cap=cap,
                          size_weight=size_weight, diagonal=True)
        if d < min_drive:
            continue
        if dx < 0:
            dl = max(dl, d)
        else:
            dr = max(dr, d)
    return dl, dr


def object_demand(y: float | None, vy: float | None, dx: float | None = None,
                  stake: float = 1.0, paddle_y: float = PADDLE_Y, k: float = PERSPECTIVE_K,
                  eps: float = URGENCY_EPS,
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
    off smoothly, which a linear readout can use.
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
        loom = (k * v / (d * d)) if v > 0.0 else 0.0
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
                       cap: float = DRIVE_CAP, steps: int = BRAIN_STEPS,
                       vx_a: float | None = None, vx_b: float | None = None,
                       vx_gain: float = 0.0) -> np.ndarray:
    """Both objects drive the *same* chase channel (LC10a) on their side, with a
    drive set by the caller. Neither pathway is privileged, so the winner is the
    object that asks for more -- feed `object_demand(y, vy, dx, stake)` and the
    more urgent object wins.

    `vx` is the object's horizontal velocity (px/frame), a visible motion. With
    `vx_gain > 0` it adds a separate drive on the side the object is *moving*
    toward, alongside the position drive on the side it *is*. This is only the
    present velocity, not an extrapolation: nothing about the future is computed.

    The default `FeatureDetectors.inject` takes only one opponent, so the drive is
    built here directly from `cells["chase"]`; the format matches what `inject`
    returns, so `brain.step` consumes it the same way.
    """
    brain.reset(seed=0)
    det = FeatureDetectors(brain)
    trace = Trace(brain, types=["descending_neuron"], tau=0.1)
    dl, dr = _sides(dx_a, drive_a, dx_b, drive_b, cap, vx_a, vx_b, vx_gain)
    inj = []
    if dl > 0:
        inj.append((det.cells["chase"]["L"], np.float32(dl)))
    if dr > 0:
        inj.append((det.cells["chase"]["R"], np.float32(dr)))
    f = None
    for _ in range(steps):
        f = trace.observe(brain.step(inject=inj))
    return f


def _sides(dx_a, drive_a, dx_b, drive_b, cap, vx_a, vx_b, vx_gain):
    """Per-side drive: each object contributes on its position side, and (with
    `vx_gain > 0`) a separate term on the side of its present horizontal motion."""
    dl = dr = 0.0
    for dx, drive in ((dx_a, drive_a), (dx_b, drive_b)):
        if dx is None or drive is None:
            continue
        d = min(max(float(drive), 0.0), cap)
        if dx < 0:
            dl += d
        else:
            dr += d
    if vx_gain:
        for vx in (vx_a, vx_b):
            if vx is None:
                continue
            m = min(abs(float(vx)) * vx_gain, cap)
            if m <= 0.0:
                continue
            if vx < 0:
                dl += m
            else:
                dr += m
    return dl, dr


class ContinuousChase:
    """A brain that is never reset: one step per game frame.

    `dn_features_chase2` resets the brain and runs 8 steps for every decision (a
    160 ms window, ~50 ms on CPU), which is too slow for a 60 fps frame, so the fly
    runs in a thread at ~19 decisions/s. Here the network keeps its recurrent state
    and advances a single step per frame (~4 ms), which fits inside a frame, so the
    fly can decide every frame at the game's own rate. The features are the same
    descending-neuron trace; only the readout differs, since it must be trained on
    this regime (see `train_continuous_readout`). Over a few thousand steps the
    network does not drift: the spike count and the trace sit on a steady level, so
    the noise does not drown the signal.
    """

    def __init__(self, brain, tau: float = 0.1, vx_gain: float = 0.0):
        self.brain = brain
        self.det = FeatureDetectors(brain)
        self.trace = Trace(brain, types=["descending_neuron"], tau=tau)
        self.vx_gain = vx_gain
        # The fly's escape neurons (DNp01), one per side. A looming threat (the
        # boss's shot) drives them directly; `escape_level` reads the reflex off
        # their trace with the reflex's own sign (away from the looming side).
        self.escape = {s: brain.groups[f"escape_{s}"] for s in "LR"}
        self._escape_slot = {s: int(self.trace.slot[i[0]]) for s, i in self.escape.items()}
        brain.reset(seed=0)

    def _inject(self, dx_a, drive_a, dx_b, drive_b, cap, vx_a=None, vx_b=None,
                escape_l: float = 0.0, escape_r: float = 0.0):
        """Both objects into the chase channel (`cells["chase"]`), by side, capped;
        the boss's shots into the escape neurons. Same layout `FeatureDetectors.inject`
        returns, so `brain.step` consumes it."""
        dl, dr = _sides(dx_a, drive_a, dx_b, drive_b, cap, vx_a, vx_b, self.vx_gain)
        inj = []
        if dl > 0:
            inj.append((self.det.cells["chase"]["L"], np.float32(dl)))
        if dr > 0:
            inj.append((self.det.cells["chase"]["R"], np.float32(dr)))
        if escape_l > 0:
            inj.append((self.escape["L"], np.float32(escape_l)))
        if escape_r > 0:
            inj.append((self.escape["R"], np.float32(escape_r)))
        return inj

    def step(self, dx_a: float | None = None, drive_a: float | None = None,
             dx_b: float | None = None, drive_b: float | None = None,
             cap: float = DRIVE_CAP, vx_a: float | None = None,
             vx_b: float | None = None, escape_l: float = 0.0,
             escape_r: float = 0.0) -> np.ndarray:
        """One step: inject the objects' drives and the escape drive, return features."""
        inj = self._inject(dx_a, drive_a, dx_b, drive_b, cap, vx_a, vx_b, escape_l, escape_r)
        return self.trace.observe(self.brain.step(inject=inj))

    def escape_level(self) -> float:
        """The escape reflex read off DNp01: how hard the fly wants to flee.

        Positive means the looming side is the right, so the fly turns left -- the
        sign is the reflex's own direction, not a trained weight. Zero when neither
        escape neuron is firing, so a run with no shots is unchanged.
        """
        t = self.trace.trace
        return float(t[self._escape_slot["R"]].sum() - t[self._escape_slot["L"]].sum())


def train_continuous_readout(brain, base: float = 0.8, cap: float = 3.0, size_weight: float = 0.1,
                             item_stake: float = 0.8, diagonal: bool = False, with_item: bool = True,
                             steps: int = 2500, burn: int = 200, seed: int = 0,
                             vx_gain: float = 0.0) -> Readout:
    """Fit the readout for `ContinuousChase` on one long, uninterrupted run.

    The brain is stepped without resetting over a slow random walk of the ball and
    the item (as in play), so the trace carries the same history it will see at run
    time; each step is labelled by the side of whichever drive is larger, as in the
    per-decision trainer. Only the sampling differs: the readout sees a sliding
    window instead of a fresh 8-step response, which is why it needs its own fit.

    This is the *chase* reflex only. The escape reflex (the boss's shots) is not
    trained here: it is read off the escape neurons with a fixed sign at run time
    (see `ContinuousChase.escape_level`), so this fit is untouched by it.
    """
    rng = np.random.default_rng(seed)
    fly = ContinuousChase(brain, vx_gain=vx_gain)
    by, bvy = 80.0, 0.0
    iy, ivy = 120.0, 0.5
    bdx = idx = 0.0
    bvx = 0.0
    have_item = with_item
    X, y = [], []
    for t in range(steps):
        prev_bdx = bdx
        bdx = float(np.clip(bdx + rng.normal(0.0, 6.0), -80.0, 80.0))
        bvx = 0.0 if t == 0 else bdx - prev_bdx
        idx = float(np.clip(idx + rng.normal(0.0, 6.0), -80.0, 80.0))
        by = float(np.clip(by + bvy, 50.0, 132.0))
        bvy = float(np.clip(bvy + rng.normal(0.0, 0.5), -2.0, 2.0))
        iy = float(np.clip(iy + ivy, 50.0, 132.0))
        ivy = float(np.clip(ivy + rng.normal(0.0, 0.2), 0.2, 1.0))
        if rng.random() < 0.02:               # start a fresh situation now and then
            by, bvy = rng.uniform(50.0, 85.0), rng.uniform(-2.0, 0.0)
            iy, ivy = rng.uniform(50.0, 132.0), rng.uniform(0.3, 1.0)
            have_item = with_item and rng.random() > 0.25
        db = base + object_demand(by, bvy, bdx, stake=STAKE_BALL, size_weight=size_weight)
        di = object_demand(iy, ivy, idx, stake=item_stake, diagonal=diagonal) if have_item else 0.0
        f = fly.step(bdx, db, idx if have_item else None, di if have_item else None,
                     cap=cap, vx_a=bvx)
        if t >= burn:
            X.append(f)
            y.append(int((idx if di > db else bdx) > 0))
    return Readout.fit(np.stack(X), np.array(y), kind="logistic")


def train_readout(brain, span: float = 80.0, samples: int = 33, verbose: bool = False) -> Readout:
    """Fit a left/right readout on a sweep of ball offsets, labelled by side.

    The label is just `dx > 0`, so the readout learns the reflex "ball on my
    right, go right". There is no reward and no learning in the connectome; the
    only trained weights are this linear map over the descending neurons.
    """
    dxs = np.linspace(-span, span, samples)
    feats = np.stack([dn_features(brain, dx) for dx in dxs])
    return Readout.fit(feats, (dxs > 0).astype(int), kind="logistic", verbose=verbose)
