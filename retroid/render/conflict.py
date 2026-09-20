"""Two-signal conflict: the ball (chase) against a falling item (shot).

A controlled stimulus-response probe, no game loop. The readout is trained only on
single-object trials (ball alone, item alone), each labelled by that object's side,
so it learns "move toward whichever object is there". Then both are injected at
once, on opposite sides, and we sweep the item's angular size (its urgency). The
question is where the decision flips from following the ball to following the item,
and whether that crossover depends on the connectome's wiring.

    python retroid/render/conflict.py --ablate none
    python retroid/render/conflict.py --ablate shuffle

Prints a table of P(right) as the item grows, for both conflict directions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

GAME = Path(__file__).resolve().parents[1]      # retroid/
ROOT = GAME.parent                              # repository root (data/, fly-ai/)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fly-ai"))
sys.path.insert(0, str(GAME))

from flybrain import FlyBrain
from retroidsim import ablate, dn_features_pair

BALL_SIZE = 16.0
ITEM_TRAIN_SIZE = 16.0
BALL_DX = 60.0
SPAN = 80.0


def train_readout(brain):
    """Fit on single-object sweeps only: ball alone, then item alone."""
    from flybrain.reservoir import Readout

    dxs = np.linspace(-SPAN, SPAN, 33)
    ball = np.stack([dn_features_pair(brain, dx, None) for dx in dxs])
    item = np.stack([dn_features_pair(brain, None, dx, size_item=ITEM_TRAIN_SIZE) for dx in dxs])
    feats = np.concatenate([ball, item])
    labels = np.concatenate([(dxs > 0).astype(int), (dxs > 0).astype(int)])
    return Readout.fit(feats, labels, kind="logistic", verbose=True)


def decision(brain, readout, dx_ball, dx_item, size_item):
    p = float(readout.predict(dn_features_pair(brain, dx_ball, dx_item, size_item=size_item)))
    return p, (1 if p >= 0.5 else -1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "fly-data"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ablate", choices=("none", "shuffle", "rewire", "silence"), default="none")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading the connectome...", flush=True)
    brain = FlyBrain(data=args.data, device=args.device)
    ablate(brain, args.ablate, args.seed)
    readout = train_readout(brain)
    print("ablation=%s  readout cross-validated AUC %.3f" % (args.ablate, readout.cv_score), flush=True)

    print("\nsanity (single object):")
    for name, db, di in (("ball +60", BALL_DX, None), ("ball -60", -BALL_DX, None),
                         ("item +60", None, BALL_DX), ("item -60", None, -BALL_DX)):
        p, d = decision(brain, readout, db, di, ITEM_TRAIN_SIZE)
        print("  %-9s -> p(right)=%.2f  side=%+d" % (name, p, d))

    print("\nconflict: ball at %+.0f, item opposite, sweep item size" % BALL_DX)
    sizes = [4, 6, 8, 10, 12, 14, 16, 20, 24, 30, 40, 55]
    curves = {}
    for ball_side in (1, -1):
        print("  ball on the %s (want %s):" % ("right" if ball_side > 0 else "left",
                                                "right" if ball_side > 0 else "left"))
        ps = []
        for s in sizes:
            p, d = decision(brain, readout, ball_side * BALL_DX, -ball_side * BALL_DX, s)
            follows = "ball" if d == ball_side else "item"
            ps.append(p)
            print("    item size %2d -> p(right)=%.2f  follows %s" % (s, p, follows))
        curves[ball_side] = np.array(ps)

    np.savez(str(GAME / "render" / ("conflict_%s.npz" % args.ablate)),
             sizes=np.array(sizes), right=curves[1], left=curves[-1], ablate=np.array(args.ablate))
    print("\nsaved", GAME / "render" / ("conflict_%s.npz" % args.ablate))


if __name__ == "__main__":
    main()
