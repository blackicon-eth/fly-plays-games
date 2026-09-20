"""Record the fly playing Retroid with two signals: the ball and a falling item.

The readout is trained only on single-object trials (ball alone, item alone), as in
`conflict.py`, so it never sees a conflict. In the game both are injected at once,
the ball into the chase channel (`opp`) and the item into the projectile channel
(`shots`), and the button is whatever the connectome's arbitration produces. When
the ball and the item are on opposite sides of the paddle, we log which one the fly
follows.

    python retroid/render/record_conflict.py --steps 4000
    python retroid/render/record_conflict.py --ablate shuffle --out render/conflict_shuffle.npz

Writes an npz with the game frames, spikes and a track of ball/paddle/item.
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

DIRIDX = {"up": 0, "down": 1, "left": 2, "right": 3}
BTNIDX = {"up": 0, "down": 1, "left": 2, "right": 3, "a": 4, "b": 5}
LAUNCH_HOLD = 10
SPAN = 80.0


def train_readout(brain) -> Readout:
    dxs = np.linspace(-SPAN, SPAN, 33)
    ball = np.stack([dn_features_pair(brain, dx, None) for dx in dxs])
    item = np.stack([dn_features_pair(brain, None, dx) for dx in dxs])
    feats = np.concatenate([ball, item])
    labels = np.concatenate([(dxs > 0).astype(int), (dxs > 0).astype(int)])
    return Readout.fit(feats, labels, kind="logistic", verbose=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=4000, help="game frames to drive")
    ap.add_argument("--rom", default=str(DEFAULT_ROM))
    ap.add_argument("--window", default="null")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--data", default=str(ROOT / "data" / "fly-data"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ablate", choices=("none", "shuffle", "rewire", "silence"), default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(GAME / "render" / "conflict_play.npz"))
    args = ap.parse_args()

    print("loading the connectome...", flush=True)
    view = FlyBrain(data=args.data, device=args.device)
    decide = FlyBrain(data=args.data, device=args.device)
    ablate(decide, args.ablate, args.seed)
    readout = train_readout(decide)
    print("ablation=%s  readout cross-validated AUC %.3f" % (args.ablate, readout.cv_score), flush=True)

    a = RetroidAdapter(rom=args.rom, window=args.window, scale=args.scale)
    a.reset_to_play()
    a.launch()

    az = view.azimuth
    src = np.linspace(-1, 1, 160)
    prev = None
    FR, F, BTN, TRACK = [], [], [], []
    held = None
    conflicts = {"ball": 0, "item": 0, "same": 0}

    def rec(name, st):
        nonlocal prev
        g = a.screen(gray=True)
        band = g.mean(0) / 255.0
        up = np.interp(az, src, band)
        ch = np.abs(up - prev) if prev is not None else np.zeros_like(up)
        prev = up
        dr = np.clip(0.45 * up + 1.6 * ch, 0, 1).astype(np.float32)
        fired = view.step(eye_drive=dr)
        FR.append(a.screen(gray=False))
        F.append(fired.astype(np.int32))
        BTN.append(BTNIDX.get(name, 6))
        TRACK.append((st.ball_x if st.ball_x is not None else -1,
                      st.ball_y if st.ball_y is not None else -1,
                      st.paddle_x if st.paddle_x is not None else -1,
                      st.item_x if st.item_x is not None else -1,
                      st.item_y if st.item_y is not None else -1))

    def set_held(name):
        nonlocal held
        if name != held:
            if held is not None:
                a.release(held)
            if name is not None:
                a.hold(name)
            held = name

    for step in range(args.steps):
        st = a.state()
        if st.in_play and not st.ball_resting:
            dx = st.dx
            p = float(readout.predict(dn_features_pair(decide, dx, st.item_dx)))
            want = "right" if p >= 0.5 else "left"
            if st.item_dx is not None:
                ball_side = 1 if dx > 0 else -1
                item_side = 1 if st.item_dx > 0 else -1
                if ball_side == item_side:
                    conflicts["same"] += 1
                elif want == ("right" if ball_side > 0 else "left"):
                    conflicts["ball"] += 1
                else:
                    conflicts["item"] += 1
                if step % 10 == 0:
                    print("  step %4d  CONFLICT ball dx=%+.0f item dx=%+.0f -> %s (p=%.2f)"
                          % (step, dx, st.item_dx, want, p), flush=True)
            set_held(want)
            rec(held, st)
            a.step(1)
        else:
            set_held(None)
            a.hold("a")
            for _ in range(LAUNCH_HOLD):
                rec("a", st)
                a.step(1)
            a.release("a")

    set_held(None)
    a.close()
    lens = np.array([len(f) for f in F])
    np.savez(args.out, frames=np.array(FR, np.uint8), fired=np.concatenate(F),
             starts=np.concatenate([[0], np.cumsum(lens)]), btns=np.array(BTN, np.int8),
             track=np.array(TRACK, np.int16), ablate=np.array(args.ablate))
    print("saved", args.out, "frames", len(FR), flush=True)
    print("CONFLICT SUMMARY ablation=%s  followed ball=%d  followed item=%d  same-side=%d"
          % (args.ablate, conflicts["ball"], conflicts["item"], conflicts["same"]), flush=True)


if __name__ == "__main__":
    main()
