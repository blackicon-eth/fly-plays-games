"""Record the fly actually driving Pokémon Red, for the "what the fly really does" clip.

Unlike render/record_journey.py (where a scripted teacher reads the RAM), here the
buttons come from the connectome: the nearest object's offset is injected into the
fly's visual channels, and a small readout on its descending neurons picks left or
right. It is a reflex with no plan, so the behaviour is aimless on purpose. The
connectome's screen-driven activity is recorded for the right-hand panel.

    python pokemon-red/render/record_fly.py --scene pallet --steps 80

Writes `pokemon-red/render/fly_drive.npz`, which render_journey.py can render.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

GAME = Path(__file__).resolve().parents[1]      # pokemon-red/
ROOT = GAME.parent                              # repository root (data/, fly-ai/)
sys.path.insert(0, str(GAME))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fly-ai"))

from flybrain import FlyBrain
from flybrain.eyes import FeatureDetectors
from flybrain.reservoir import Readout, Trace
from pokesim import PokemonAdapter
from pokesim.encoder_b import target as oracle_target

MOVE_FRAMES = 16
SETTLE_FRAMES = 2
OBJECT_SIZE = 16.0
DEAD_ZONE = 8.0
BRAIN_STEPS = 8
DIRIDX = {"up": 0, "down": 1, "left": 2, "right": 3}
BTNIDX = {"up": 0, "down": 1, "left": 2, "right": 3, "a": 4, "b": 5}


def dn_features(brain, dx: float, steps: int = BRAIN_STEPS) -> np.ndarray:
    brain.reset(seed=0)
    detectors = FeatureDetectors(brain)
    trace = Trace(brain, types=["descending_neuron"], tau=0.1)
    f = None
    for _ in range(steps):
        f = trace.observe(brain.step(inject=detectors.inject(opp=(float(dx), OBJECT_SIZE), threat=0.0)))
    return f


def train_readout(brain) -> Readout:
    dxs = np.linspace(-80.0, 80.0, 33)
    feats = np.stack([dn_features(brain, dx) for dx in dxs])
    side = (dxs > 0).astype(int)
    return Readout.fit(feats, side, kind="logistic", verbose=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="pallet")
    ap.add_argument("--steps", type=int, default=80, help="how many decisions to run")
    ap.add_argument("--window", default="null")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--data", default=str(ROOT / "data" / "fly-data"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(GAME / "render" / "fly_drive.npz"))
    args = ap.parse_args()

    print("loading the connectome...", flush=True)
    view = FlyBrain(data=args.data, device=args.device)      # screen-driven, for the right panel
    decide = FlyBrain(data=args.data, device=args.device)    # reset per decision, for the readout
    readout = train_readout(decide)
    print("readout: cross-validated AUC %.3f" % readout.cv_score, flush=True)

    a = PokemonAdapter(window=args.window, scale=args.scale)
    a.load_scene(args.scene)
    print("scene", args.scene, "map", a.read("map"), "pos", (a.read("x"), a.read("y")), flush=True)

    az = view.azimuth
    src = np.linspace(-1, 1, 160)
    prev = None
    FR, F, DIRS, PXY, BTN, MAPS = [], [], [], [], [], []

    def rec(name):
        nonlocal prev
        rgb = a.screen(gray=False)
        g = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        band = g[40:104].mean(0) / 255.0
        up = np.interp(az, src, band)
        ch = np.abs(up - prev) if prev is not None else np.zeros_like(up)
        prev = up
        dr = np.clip(0.45 * up + 1.6 * ch, 0, 1).astype(np.float32)
        fired = view.step(eye_drive=dr)
        FR.append(rgb)
        F.append(fired.astype(np.int32))
        DIRS.append(DIRIDX.get(name, 3))
        PXY.append((a.read("x"), a.read("y")))
        BTN.append(BTNIDX.get(name, 6))
        MAPS.append(a.read("map"))

    def press(button, frames):
        a.pb.button_press(button)
        for _ in range(frames):
            a.step(1)
            rec(button)
        a.pb.button_release(button)

    for step in range(args.steps):
        t = oracle_target(a)
        if t is None:
            button = "a"
        else:
            p = float(readout.predict(dn_features(decide, t.dx)))
            button = "right" if p >= 0.5 else "left"
            if step % 5 == 0:
                print("step %4d  dx=%+5.0f  p(right)=%.2f -> %s" % (step, t.dx, p, button), flush=True)
        press(button, MOVE_FRAMES)
        a.step(SETTLE_FRAMES)

    a.close()
    lens = np.array([len(f) for f in F])
    np.savez(args.out, frames=np.array(FR, np.uint8), fired=np.concatenate(F),
             starts=np.concatenate([[0], np.cumsum(lens)]), dirs=np.array(DIRS, np.int8),
             pxy=np.array(PXY, np.int16), btns=np.array(BTN, np.int8), maps=np.array(MAPS, np.int16))
    print("saved", args.out, "frames", len(FR), flush=True)


if __name__ == "__main__":
    main()
