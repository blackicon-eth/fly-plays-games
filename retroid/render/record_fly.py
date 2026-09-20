"""Record the fly actually driving Retroid, and score it.

The ball's horizontal offset is injected into the fly's visual channels, a small
readout on its descending neurons picks left or right, and the paddle follows.
As everywhere in this project the position of the ball is handed to the visual
front end, and the connectome only does visual projection neurons -> descending
neurons. The task is a single reflex with no memory, so the honest question is
not "can it play" but "does the connectome matter": `--ablate shuffle` scrambles
the connectome's weights and retrains the readout, and the score should collapse.

    python retroid/render/record_fly.py --steps 1200
    python retroid/render/record_fly.py --ablate shuffle --out retroid/render/ablated.npz

Writes an npz with the game frames, the connectome spikes and the paddle/ball
track, plus a printed score (frames survived, mean |dx|).
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
from retroidsim import DEFAULT_ROM, RetroidAdapter, ablate, dn_features, train_readout

DIRIDX = {"up": 0, "down": 1, "left": 2, "right": 3}
BTNIDX = {"up": 0, "down": 1, "left": 2, "right": 3, "a": 4, "b": 5}
LAUNCH_HOLD = 10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=1200, help="game frames to drive")
    ap.add_argument("--rom", default=str(DEFAULT_ROM))
    ap.add_argument("--window", default="null")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--data", default=str(ROOT / "data" / "fly-data"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ablate", choices=("none", "shuffle", "rewire", "silence"), default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(GAME / "render" / "fly_drive.npz"))
    args = ap.parse_args()

    print("loading the connectome...", flush=True)
    view = FlyBrain(data=args.data, device=args.device)      # screen-driven, for the panel
    decide = FlyBrain(data=args.data, device=args.device)    # reset per decision, for the readout
    ablate(decide, args.ablate, args.seed)
    readout = train_readout(decide, verbose=True)
    print("ablation=%s  readout cross-validated AUC %.3f" % (args.ablate, readout.cv_score), flush=True)

    a = RetroidAdapter(rom=args.rom, window=args.window, scale=args.scale)
    a.reset_to_play()
    a.launch()

    az = view.azimuth
    src = np.linspace(-1, 1, 160)
    prev = None
    FR, F, DIRS, BTN, TRACK = [], [], [], [], []
    held = None

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
        DIRS.append(DIRIDX.get(name, 3))
        BTN.append(BTNIDX.get(name, 6))
        TRACK.append((st.ball_x if st.ball_x is not None else -1,
                      st.ball_y if st.ball_y is not None else -1,
                      st.paddle_x if st.paddle_x is not None else -1))

    def set_held(name):
        nonlocal held
        if name != held:
            if held is not None:
                a.release(held)
            if name is not None:
                a.hold(name)
            held = name

    survived, dxs = 0, []
    for step in range(args.steps):
        st = a.state()
        if st.in_play and not st.ball_resting:
            dx = st.dx
            dxs.append(abs(dx))
            survived += 1
            p = float(readout.predict(dn_features(decide, dx)))
            set_held("right" if p >= 0.5 else "left")
            rec(held, st)
            a.step(1)
            if step % 100 == 0:
                print("step %4d  dx=%+5.0f  p(right)=%.2f -> %s" % (step, dx, p, held), flush=True)
        else:
            set_held(None)
            a.hold("a")
            for _ in range(LAUNCH_HOLD):
                rec("a", st)
                a.step(1)
            a.release("a")
            print("step %4d  ball lost; relaunching" % step, flush=True)

    set_held(None)
    a.close()
    lens = np.array([len(f) for f in F])
    np.savez(args.out, frames=np.array(FR, np.uint8), fired=np.concatenate(F),
             starts=np.concatenate([[0], np.cumsum(lens)]), dirs=np.array(DIRS, np.int8),
             btns=np.array(BTN, np.int8), track=np.array(TRACK, np.int16),
             ablate=np.array(args.ablate))
    mean_dx = float(np.mean(dxs)) if dxs else float("nan")
    print("saved", args.out, "frames", len(FR), flush=True)
    print("SCORE ablation=%s frames_in_play=%d mean|dx|=%.2f" % (args.ablate, survived, mean_dx), flush=True)


if __name__ == "__main__":
    main()
