"""Bookmark a few mid-flight moments of level 1 as extra starting scenes.

`level1` is the level start (ball resting on the paddle). To test the fly from
more than one starting condition we play a game with the oracle tracker and save
snapshots while the ball is in play, so each scene is a different ball position
and direction. Scenes are gitignored (`retroid/roms/scenes/`).

    python retroid/render/make_scenes.py --every 450
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

GAME = Path(__file__).resolve().parents[1]      # retroid/
ROOT = GAME.parent                              # repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GAME))

from retroidsim import DEFAULT_ROM, RetroidAdapter


def oracle_act(st):
    dx = st.dx
    if dx is None:
        return None
    if dx > 2.0:
        return "right"
    if dx < -2.0:
        return "left"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="level1", help="scene to start from")
    ap.add_argument("--names", nargs="*", default=["level1_b", "level1_c", "level1_d"])
    ap.add_argument("--every", type=int, default=450, help="save a scene every N in-play frames")
    ap.add_argument("--max-frames", type=int, default=6000)
    ap.add_argument("--rom", default=str(DEFAULT_ROM))
    args = ap.parse_args()

    a = RetroidAdapter(rom=args.rom)
    if not a.has_scene(args.base):
        a.reset_to_play(name=args.base, force=True)
    a.load_scene(args.base)
    a.launch()
    for _ in range(90):
        a.step(1)
        if a.ball_live():
            break

    saved, held, live = 0, None, 0
    for _ in range(args.max_frames):
        if a.ball() is None:
            print("ball lost before all scenes were saved", flush=True)
            break
        if saved < len(args.names) and live and live % args.every == 0:
            a.save_scene(args.names[saved])
            st = a.state()
            print(f"saved {args.names[saved]}  ball=({st.ball_x},{st.ball_y})  paddle={st.paddle_x:.0f}",
                  flush=True)
            saved += 1
        st = a.state()
        want = oracle_act(st)
        if want != held:
            if held:
                a.release(held)
            if want:
                a.hold(want)
            held = want
        a.step(1)
        live += 1
    if held:
        a.release(held)
    a.close()
    print(f"saved {saved}/{len(args.names)} scenes in {a.scenes_dir}", flush=True)


if __name__ == "__main__":
    main()
