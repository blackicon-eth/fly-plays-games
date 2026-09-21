"""Paired comparison: the real connectome and a shuffled one, on the *same* states.

`record_conflict.py` compares two separate runs, but each run's decisions change
the game, so the two trajectories diverge and the conflict states differ. Here one
run is driven by the real connectome while, at every frame, *both* readouts are
evaluated on the identical (ball dx, item dx). The comparison is therefore paired:
same states, two decoders.

The default channels (ball -> `opp`/LC10a, item -> `shots`/LPLC1) are not
comparable: the readout's intercept, not the objects, decides the conflict. Three
modes probe around that:

* `--innate` replaces the trained readout with a fixed left-minus-right DN sum, so
  the drive's *magnitude* survives and the connectome arbitrates by its own wiring.
* `--chase2` routes *both* objects through the chase channel, so neither pathway is
  privileged.
* `--urgency` sets each object's drive from `object_demand` (proximity x imminence,
  zero while ascending), so the more urgent object wins.

The summary reports, over conflict frames, how often each decoder follows the item,
how often they disagree, the split by ball demand, and how many falling items the
fly actually caught.

    python retroid/render/conflict_paired.py --steps 2500
    python retroid/render/conflict_paired.py --steps 2500 --innate
    python retroid/render/conflict_paired.py --steps 2500 --chase2 --urgency
    python retroid/render/conflict_paired.py --chase2 --urgency --scene level1_c --solo
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
from retroidsim import (DEFAULT_ROM, STAKE_BALL, STAKE_ITEM, RetroidAdapter, ablate,
                        dn_features_chase2, dn_features_pair, object_demand, proximity_size)

LAUNCH_HOLD = 10
SPAN = 80.0
OBJECT_SIZE = 16.0
SIZES = (8.0, 16.0, 24.0, 32.0)   # training sweep when urgency varies the size


def train_readout(brain, swap: bool = False, sizes=(OBJECT_SIZE,)) -> Readout:
    dxs = np.linspace(-SPAN, SPAN, 33)
    X, y = [], []
    for s in sizes:
        for dx in dxs:
            if swap:   # ball drives the shot channel, item the chase channel
                ball = dn_features_pair(brain, None, dx, size_item=s)
                item = dn_features_pair(brain, dx, None, size_ball=s)
            else:
                ball = dn_features_pair(brain, dx, None, size_ball=s)
                item = dn_features_pair(brain, None, dx, size_item=s)
            X += [ball, item]
            y += [int(dx > 0), int(dx > 0)]
    return Readout.fit(np.stack(X), np.array(y), kind="logistic")


def bucket(y: int | None) -> str:
    if y is None:
        return "unknown"
    if y >= 100:
        return "low (urgent)"
    if y <= 70:
        return "high (safe)"
    return "mid"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--rom", default=str(DEFAULT_ROM))
    ap.add_argument("--data", default=str(ROOT / "data" / "fly-data"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swap", action="store_true",
                    help="control: ball drives the shot channel and item the chase channel")
    ap.add_argument("--urgency", action="store_true",
                    help="size each object from its screen y (loom grows as it falls)")
    ap.add_argument("--innate", action="store_true",
                    help="fixed opponent decoder (left DNs minus right DNs), no trained readout")
    ap.add_argument("--chase2", action="store_true",
                    help="both objects drive the chase channel (LC10a), drive ~ size; implies --innate")
    ap.add_argument("--scene", default=None,
                    help="start from a saved mid-flight scene (e.g. level1_b) instead of the level start")
    ap.add_argument("--solo", action="store_true",
                    help="skip the shuffled connectome (about half the cost; no shuffle comparison)")
    args = ap.parse_args()
    innate = args.innate or args.chase2

    sizes = SIZES if args.urgency else (OBJECT_SIZE,)
    print("loading connectome(s)...", flush=True)
    real = FlyBrain(data=args.data, device=args.device)
    shuf = None
    if not args.solo:
        shuf = FlyBrain(data=args.data, device=args.device)
        ablate(shuf, "shuffle", args.seed)
    if innate:
        r_real = r_shuf = None
        dn = real.cells(["descending_neuron"])
        sd = np.asarray(real.side)[dn].astype(str)
        Lm, Rm = sd == "L", sd == "R"
        print("decoder=innate (L-R DNs, %d L / %d R)  chase2=%s  urgency=%s  solo=%s"
              % (int(Lm.sum()), int(Rm.sum()), args.chase2, args.urgency, args.solo), flush=True)
    else:
        r_real = train_readout(real, args.swap, sizes)
        r_shuf = None if shuf is None else train_readout(shuf, args.swap, sizes)
        print("decoder=trained  channels=%s  urgency=%s  AUC real %.3f  shuffle %s"
              % ("swapped" if args.swap else "ball=opp item=shots", args.urgency,
                 r_real.cv_score, "n/a" if r_shuf is None else "%.3f" % r_shuf.cv_score), flush=True)

    def decide_innate(f):
        return 1 if float(f[Rm].sum()) > float(f[Lm].sum()) else -1

    a = RetroidAdapter(rom=args.rom, window="null")
    if args.scene:
        a.load_scene(args.scene)
        a.step(6)   # prime the ball-motion history for `ball_live`
        print("scene: %s" % args.scene, flush=True)
    else:
        a.reset_to_play()
        a.launch()

    def feats(brain, st, ball_dx, item_dx, ball_vy, item_vy):
        if args.chase2:
            if args.urgency:
                db = object_demand(st.ball_y, ball_vy, ball_dx, stake=STAKE_BALL)
                di = object_demand(st.item_y, item_vy, item_dx, stake=STAKE_ITEM)
            else:
                db = di = 1.0
            return dn_features_chase2(brain, ball_dx, db, item_dx, di)
        if args.urgency:
            sb, si = proximity_size(st.ball_y), proximity_size(st.item_y)
        else:
            sb = si = OBJECT_SIZE
        if args.swap:
            return dn_features_pair(brain, item_dx, ball_dx, size_ball=si, size_item=sb)
        return dn_features_pair(brain, ball_dx, item_dx, size_ball=sb, size_item=si)

    held = None
    prev_ball_y = prev_item_y = None
    conflict = 0
    real_item = shuf_item = disagree = 0
    buckets: dict[str, list[int]] = {}
    records: list[tuple[float, int]] = []   # (ball demand, followed item) for the demand split
    items_caught = items_missed = items_gap = item_frames = 0
    prev_item = None
    item_hist: list[tuple[int, int]] = []
    for _ in range(args.steps):
        st = a.state()
        ball_vy = 0.0 if (st.ball_y is None or prev_ball_y is None) else st.ball_y - prev_ball_y
        item_vy = 0.0 if (st.item_y is None or prev_item_y is None) else st.item_y - prev_item_y
        prev_ball_y, prev_item_y = st.ball_y, st.item_y
        cur_item = None if st.item_x is None else (st.item_x, st.item_y)
        if cur_item is not None:
            item_frames += 1
            item_hist.append(cur_item)
            item_hist = item_hist[-8:]
        if prev_item is not None and cur_item is None:
            px, py = prev_item
            under = st.paddle_x is not None and abs(px - st.paddle_x) <= 12
            if py >= 128 and under:
                items_caught += 1
                verdict = "caught"
            elif py >= 128:
                items_missed += 1
                verdict = "missed"
            else:
                items_gap += 1
                verdict = "gap (ignored)"
            print("    item drop: last y=%d x=%d paddle=%.0f -> %s   traj=%s"
                  % (py, px, st.paddle_x if st.paddle_x is not None else -1, verdict,
                     " ".join("%d,%d" % p for p in item_hist)), flush=True)
        prev_item = cur_item
        if a.ball_live():
            dx, idx = st.dx, st.item_dx
            if innate:
                da = decide_innate(feats(real, st, dx, idx, ball_vy, item_vy))
                db = da if shuf is None else decide_innate(feats(shuf, st, dx, idx, ball_vy, item_vy))
            else:
                da = 1 if float(r_real.predict(feats(real, st, dx, idx, ball_vy, item_vy))) >= 0.5 else -1
                db = da if shuf is None else (1 if float(r_shuf.predict(feats(shuf, st, dx, idx, ball_vy, item_vy))) >= 0.5 else -1)
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
                    b = buckets.setdefault(bucket(st.ball_y), [0, 0])
                    b[0] += 1
                    b[1] += int(da != ball_side)
                    if args.chase2 and args.urgency:
                        records.append((object_demand(st.ball_y, ball_vy, dx, stake=STAKE_BALL),
                                        int(da != ball_side)))
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
    print("conflict frames: %d   (channels=%s urgency=%s)"
          % (conflict, "swapped" if args.swap else "ball=opp item=shots", args.urgency), flush=True)
    print("  real    followed the item: %d (%.0f%%)" % (real_item, 100 * real_item / max(conflict, 1)), flush=True)
    print("  shuffle followed the item: %d (%.0f%%)" % (shuf_item, 100 * shuf_item / max(conflict, 1)), flush=True)
    print("  the two readouts disagreed on %d (%.0f%%) of conflict frames"
          % (disagree, 100 * disagree / max(conflict, 1)), flush=True)
    print("  items on screen %d frames; caught %d, missed %d, gaps %d (of %d drops)"
          % (item_frames, items_caught, items_missed, items_gap,
             items_caught + items_missed + items_gap), flush=True)
    if args.urgency:
        print("  by ball urgency (real readout):", flush=True)
        for name in ("low (urgent)", "mid", "high (safe)", "unknown"):
            if name in buckets:
                n, it = buckets[name]
                print("    %-12s n=%4d  follows item %3.0f%%" % (name, n, 100 * it / max(n, 1)), flush=True)
    if records:
        records.sort(key=lambda r: r[0])
        k = max(1, len(records) // 3)
        print("  by ball demand (real readout, terciles of object_demand):", flush=True)
        for label, lo in (("low", 0), ("mid", k), ("high", 2 * k)):
            chunk = records[lo:] if label == "high" else records[lo:lo + k]
            if not chunk:
                continue
            it = sum(r[1] for r in chunk)
            print("    %-4s n=%4d  demand %.2f..%.2f  follows item %3.0f%%"
                  % (label, len(chunk), chunk[0][0], chunk[-1][0], 100 * it / len(chunk)), flush=True)


if __name__ == "__main__":
    main()
