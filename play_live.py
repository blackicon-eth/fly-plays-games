"""Watch Pokémon Red run with the fly connectome in the loop, in a PyBoy window.

What you are watching is the honest version of "the fly plays": the nearest map
object is turned into visual-projection drive (LPLC2 / LC4 / LPLC1 / LC10a), the
spikes go through the frozen MaleCNS connectome, and a small readout on its
descending neurons picks left or right. The readout is trained at startup on a
synthetic sweep of offsets, so it needs nothing but the brain files. It is a
reflex, not a plan -- the long walk from Pallet to Viridian in the video is a
separate, scripted teacher (see render/record_journey.py, which can also open a
window with --window SDL2).

    python play_live.py --scene route1 --scale 4
    python play_live.py --make-scene overworld   # if you have no saved scenes

Run it from the repository root. Scene bookmarks live in roms/scenes/ and are not
committed; the connectome files are downloaded to --data on first use.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "fly-ai")

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


def dn_features(brain, dx: float, steps: int = BRAIN_STEPS) -> np.ndarray:
    """Drive the visual channels with one offset, return the descending-neuron trace."""
    brain.reset(seed=0)
    detectors = FeatureDetectors(brain)
    trace = Trace(brain, types=["descending_neuron"], tau=0.1)
    f = None
    for _ in range(steps):
        f = trace.observe(brain.step(inject=detectors.inject(opp=(float(dx), OBJECT_SIZE), threat=0.0)))
    return f


def train_readout(brain, steps: int = BRAIN_STEPS) -> Readout:
    """A logistic readout on the connectome, fit on a synthetic left/right sweep."""
    dxs = np.linspace(-80.0, 80.0, 33)
    feats = np.stack([dn_features(brain, dx, steps) for dx in dxs])
    side = (dxs > 0).astype(int)
    return Readout.fit(feats, side, kind="logistic", verbose=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="scene bookmark in roms/scenes/ (default: pallet, then route1, then overworld)")
    ap.add_argument("--make-scene", default=None, metavar="NAME", help="bookmark the current moment as NAME and exit")
    ap.add_argument("--scale", type=int, default=4, help="PyBoy window scale")
    ap.add_argument("--steps", type=int, default=400, help="how many decisions to run")
    ap.add_argument("--data", default="data/fly-data", help="folder with brain.npz and weights.npz")
    ap.add_argument("--device", default="cpu", help="cpu, cuda or auto")
    args = ap.parse_args()

    if args.make_scene:
        a = PokemonAdapter(window="SDL2", scale=args.scale)
        a.reset_to_overworld()
        path = a.save_scene(args.make_scene)
        print("saved", path)
        a.close()
        return

    print("loading the connectome...", flush=True)
    brain = FlyBrain(data=args.data, device=args.device)
    readout = train_readout(brain)
    print(f"readout: cross-validated AUC {readout.cv_score:.3f} on a synthetic L/R sweep", flush=True)

    a = PokemonAdapter(window="SDL2", scale=args.scale)
    scene = args.scene
    if scene is None:
        for name in ("pallet", "route1"):
            if a.has_scene(name):
                scene = name
                break
    if scene:
        a.load_scene(scene)
    else:
        a.reset_to_overworld()
    print("scene:", scene or "overworld", "| map", a.read("map"),
          "| pos", (a.read("x"), a.read("y")), flush=True)
    print("left/right are chosen by the connectome. Ctrl-C to stop.", flush=True)

    try:
        for step in range(args.steps):
            t = oracle_target(a)
            if t is None or abs(t.dx) < DEAD_ZONE:
                a.press("a", 6)
                continue
            p = float(readout.predict(dn_features(brain, t.dx)))
            button = "right" if p >= 0.5 else "left"
            a.press(button, MOVE_FRAMES)
            a.step(SETTLE_FRAMES)
            if step % 5 == 0:
                fired = int(np.asarray(brain.fired).size)
                print(f"  step {step:4d}  dx={t.dx:+5.0f}  p(right)={p:.2f} -> {button}"
                      f"  | DN spikes {fired}", flush=True)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        a.close()


if __name__ == "__main__":
    main()
