"""Paired comparison: the real connectome and a shuffled one, on the *same* states.

`record_conflict.py` compares two separate runs, but each run's decisions change
the game, so the two trajectories diverge and the conflict states differ. Here one
run is driven by the real connectome while, at every frame, *both* readouts are
evaluated on the identical (ball dx, item dx). The comparison is therefore paired:
same states, two decoders.

    python retroid/render/conflict_paired.py --steps 2500

Prints, over conflict frames, how often each readout follows the ball or the item,
and how often they disagree.
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
from flybrain.reservoir import Readout
from retroidsim import DEFAULT_ROM, RetroidAdapter, ablate, dn_features_pair

LAUNCH_HOLD = 10
SPAN = 80.0


def train_readout(brain) -> Readout:
    dxs = np.linspace(-SPAN, SPAN, 33)
    ball = np.stack([dn_features_pair(brain, dx, None) for dx in dxs])
    item = np.stack([dn_features_pair(brain, None, dx) for dx in dxs])
    feats = np.concatenate([ball, item])
    labels = np.concatenate([(dxs > 0).astype(int), (dxs > 0).astype(int)])
    return Readout.fit(feats, labels, kind="logistic")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--rom", default=str(DEFAULT_ROM))
    ap.add_argument("--data", default=str(ROOT / "data" / "fly-data"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading two connectomes (real + shuffled)...", flush=True)
    real = FlyBrain(data=args.data, device=args.device)
    shuf = FlyBrain(data=args.data, device=args.device)
    ablate(shuf, "shuffle", args.seed)
    r_real = train_readout(real)
    r_shuf = train_readout(shuf)
    print("AUC real %.3f   AUC shuffle %.3f" % (r_real.cv_score, r_shuf.cv_score), flush=True)

    a = RetroidAdapter(rom=args.rom, window="null")
    a.reset_to_play()
    a.launch()

    held = None
    conflict = 0
    real_item = shuf_item = 0
    disagree = 0
    for step in range(args.steps):
        st = a.state()
        if st.in_play and not st.ball_resting:
            dx, idx = st.dx, st.item_dx
            pa = float(r_real.predict(dn_features_pair(real, dx, idx)))
            pb = float(r_shuf.predict(dn_features_pair(shuf, dx, idx)))
            da = 1 if pa >= 0.5 else -1
            db = 1 if pb >= 0.5 else -1
            want = "right" if da > 0 else "left"
            if idx is not None:
                ball_side = 1 if dx > 0 else -1
                item_side = 1 if idx > 0 else -1
                if ball_side != item_side:
                    conflict += 1
                    if da != ball_side:
                        real_item += 1
                    if db != ball_side:
                        shuf_item += 1
                    if da != db:
                        disagree += 1
            if want != held:
                if held:
                    a.release(held)
                a.hold(want)
                held = want
            a.step(1)
        else:
            if held:
                a.release(held)
                held = None
            a.hold("a")
            a.step(LAUNCH_HOLD)
            a.release("a")

    if held:
        a.release(held)
    a.close()
    print("conflict frames: %d" % conflict, flush=True)
    print("  real   followed the item: %d (%.0f%%)" % (real_item, 100 * real_item / max(conflict, 1)), flush=True)
    print("  shuffle followed the item: %d (%.0f%%)" % (shuf_item, 100 * shuf_item / max(conflict, 1)), flush=True)
    print("  the two readouts disagreed on %d (%.0f%%) of conflict frames"
          % (disagree, 100 * disagree / max(conflict, 1)), flush=True)


if __name__ == "__main__":
    main()
