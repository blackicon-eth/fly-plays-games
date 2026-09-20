"""Watch Retroid run with the fly connectome in the loop, in a PyBoy window.

The ball's horizontal offset is turned into visual-projection drive (LPLC2 / LC4
/ LPLC1 / LC10a), the spikes go through the frozen MaleCNS connectome, and a
small readout on its descending neurons holds left or right. The readout is
trained at startup on a synthetic sweep of offsets, so it needs nothing but the
brain files. It is one reflex with no memory: keep the paddle under the ball.

    python retroid/play_live.py --scale 4
    python retroid/play_live.py --make-scene level1   # bookmark the level start

Run it from anywhere. The ROM goes in `retroid/roms/Retroid.gb` and is not
committed; scene bookmarks live in `retroid/roms/scenes/`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

GAME = Path(__file__).resolve().parent          # retroid/
ROOT = GAME.parent                              # repository root (data/, fly-ai/)
sys.path.insert(0, str(GAME))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fly-ai"))

from flybrain import FlyBrain
from retroidsim import DEFAULT_ROM, RetroidAdapter, dn_features, train_readout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-scene", default=None, metavar="NAME", help="bookmark the level start as NAME and exit")
    ap.add_argument("--rom", default=str(DEFAULT_ROM))
    ap.add_argument("--scale", type=int, default=4, help="PyBoy window scale")
    ap.add_argument("--steps", type=int, default=2000, help="how many frames to run")
    ap.add_argument("--data", default=str(ROOT / "data" / "fly-data"), help="folder with brain.npz and weights.npz")
    ap.add_argument("--device", default="cpu", help="cpu, cuda or auto")
    args = ap.parse_args()

    if args.make_scene:
        a = RetroidAdapter(rom=args.rom, window="SDL2", scale=args.scale)
        a.reset_to_play(name=args.make_scene, force=True)
        print("saved", a.scene_path(args.make_scene))
        a.close()
        return

    print("loading the connectome...", flush=True)
    brain = FlyBrain(data=args.data, device=args.device)
    readout = train_readout(brain)
    print(f"readout: cross-validated AUC {readout.cv_score:.3f} on a synthetic L/R sweep", flush=True)

    a = RetroidAdapter(rom=args.rom, window="SDL2", scale=args.scale)
    a.reset_to_play()
    a.launch()
    print("level 1 started. left/right are chosen by the connectome. Ctrl-C to stop.", flush=True)

    held = None
    try:
        for step in range(args.steps):
            st = a.state()
            if not st.in_play or st.ball_resting:
                if held:
                    a.release(held)
                    held = None
                a.press("a", 10)
                continue
            dx = st.dx
            p = float(readout.predict(dn_features(brain, dx)))
            want = "right" if p >= 0.5 else "left"
            if want != held:
                if held:
                    a.release(held)
                a.hold(want)
                held = want
            a.step(1)
            if step % 30 == 0:
                fired = int(brain.fired.size)
                print(f"  step {step:4d}  ball x={st.ball_x:3d}  paddle x={st.paddle_x:5.1f}"
                      f"  dx={dx:+5.0f}  p(right)={p:.2f} -> {want}  | DN spikes {fired}", flush=True)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if held:
            a.release(held)
        a.close()


if __name__ == "__main__":
    main()
