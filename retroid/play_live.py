"""Watch Retroid run with the fly connectome in the loop, in a PyBoy window.

The ball's horizontal offset is turned into visual-projection drive (LPLC2 / LC4
/ LPLC1 / LC10a), the spikes go through the frozen MaleCNS connectome, and a
small readout on its descending neurons holds left or right. The readout is
trained at startup on a synthetic sweep of offsets, so it needs nothing but the
brain files. It is one reflex with no memory: keep the paddle under the ball.

A decision costs 8 connectome steps (~50 ms on CPU), so the brain decides ~19
times a second while the Game Boy runs at ~60 fps. The brain therefore runs in a
background thread and the emulator ticks at its own rate, reading the latest
decision: the game keeps 60 fps and smooth audio, and the paddle follows updates
that refresh every few frames.

    python retroid/play_live.py --scale 4
    python retroid/play_live.py --make-scene level1   # bookmark the level start

Run it from anywhere. The ROM goes in `retroid/roms/Retroid.gb` and is not
committed; scene bookmarks live in `retroid/roms/scenes/`.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

GAME = Path(__file__).resolve().parent          # retroid/
ROOT = GAME.parent                              # repository root (data/, fly-ai/)
sys.path.insert(0, str(GAME))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fly-ai"))

from flybrain import FlyBrain
from flybrain.reservoir import Readout
from retroidsim import (DEFAULT_ROM, STAKE_BALL, STAKE_ITEM, ContinuousChase, RetroidAdapter,
                        dn_features, dn_features_chase2, object_demand,
                        train_continuous_readout, train_readout)


def train_items_readout(brain, base: float = 0.8, cap: float = 3.0, size_weight: float = 0.1,
                        item_stake: float = 0.8, diagonal: bool = False,
                        samples: int = 220, seed: int = 0):
    """Fit a readout on the shared-chase trace to follow the *more urgent* object.

    The ball's drive is `base + object_demand(...)` and the item's is just its
    `object_demand`, both into LC10a (`dn_features_chase2`). The constant `base`
    keeps the ball's *position* signal loud and stable even when the ball is safe
    (its urgency alone collapses high up, and a weak signal lets the connectome's
    left/right asymmetry decide). `size_weight` keeps the ball's urgency about
    *imminence* rather than proximity, so a ball that is still close but moving away
    stops looking urgent and the item can win. Urgency is the separate term that lets
    the item compete: it wins only when it asks for more than `base` plus the ball's
    urgency. The label is the side of whichever drive is larger (the ball wins ties
    and the empty case). A trained readout absorbs the connectome's asymmetry --
    a rightward chase barely moves the descending neurons at low drive.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(samples):
        if rng.random() < 0.5:                      # ball safe: high up or rising
            by, bvy = rng.uniform(50.0, 85.0), rng.uniform(-2.0, 0.0)
        else:                                       # ball urgent: low and falling
            by, bvy = rng.uniform(95.0, 132.0), rng.uniform(0.5, 2.0)
        bdx = rng.uniform(-70.0, 70.0)
        db = base + object_demand(by, bvy, bdx, stake=STAKE_BALL, size_weight=size_weight)
        if rng.random() < 0.2:                      # no item on screen
            idx, di = None, 0.0
        else:
            if rng.random() < 0.5:                  # item far away / rising
                iy, ivy = rng.uniform(50.0, 95.0), rng.uniform(-1.0, 0.0)
            else:                                   # item close and falling
                iy, ivy = rng.uniform(110.0, 132.0), rng.uniform(0.3, 1.0)
            idx = rng.uniform(-70.0, 70.0)
            di = object_demand(iy, ivy, idx, stake=item_stake, diagonal=diagonal)
        f = dn_features_chase2(brain, bdx, db, idx, di, cap=cap)
        side = (idx if di > db else bdx)
        X.append(f)
        y.append(int(side > 0))
    return Readout.fit(np.stack(X), np.array(y), kind="logistic")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-scene", default=None, metavar="NAME", help="bookmark the level start as NAME and exit")
    ap.add_argument("--rom", default=str(DEFAULT_ROM))
    ap.add_argument("--scale", type=int, default=4, help="PyBoy window scale")
    ap.add_argument("--steps", type=int, default=2000, help="how many frames to run")
    ap.add_argument("--items", action="store_true",
                    help="also inject the falling item and chase it when the ball is safe")
    ap.add_argument("--base", type=float, default=0.8,
                    help="constant tracking drive added to the ball (keeps its position signal loud)")
    ap.add_argument("--cap", type=float, default=3.0, help="most drive any one side can receive")
    ap.add_argument("--size-weight", type=float, default=0.1,
                    help="how much raw proximity counts in the ball's urgency (loom dominates)")
    ap.add_argument("--item-stake", type=float, default=0.8,
                    help="how much the item's demand counts (bigger = the item wins from higher up)")
    ap.add_argument("--item-diagonal", action="store_true",
                    help="item distance = straight line paddle-item, so a far one looms less")
    ap.add_argument("--volume", type=int, default=40, help="game sound volume, 0-100")
    ap.add_argument("--exit-after", type=int, default=24,
                    help="frames without a live ball after which the level is assumed cleared and the fly drives right")
    ap.add_argument("--continuous", action="store_true",
                    help="never reset the brain: one connectome step per frame (fits 60 fps) instead of the threaded 8-step decisions")
    ap.add_argument("--scene", default=None, help="start from a saved mid-flight scene (e.g. level1_c)")
    ap.add_argument("--headless", action="store_true", help="run without a window (tests); throttled to --fps")
    ap.add_argument("--fps", type=float, default=60.0,
                    help="frame rate to target (the brain decides ~19 times a second)")
    ap.add_argument("--data", default=str(ROOT / "data" / "fly-data"), help="folder with brain.npz and weights.npz")
    ap.add_argument("--device", default="cpu", help="cpu, cuda or auto")
    args = ap.parse_args()

    if args.make_scene:
        a = RetroidAdapter(rom=args.rom, window="SDL2", scale=args.scale, sound_volume=args.volume)
        a.reset_to_play(name=args.make_scene, force=True)
        print("saved", a.scene_path(args.make_scene))
        a.close()
        return

    print("loading the connectome...", flush=True)
    brain = FlyBrain(data=args.data, device=args.device)
    fly = None
    if args.continuous:
        readout = train_continuous_readout(brain, base=args.base, cap=args.cap,
                                           size_weight=args.size_weight, item_stake=args.item_stake,
                                           diagonal=args.item_diagonal, with_item=args.items)
        fly = ContinuousChase(brain)     # fresh state for the live run
        print(f"decoder: continuous brain, one step per frame (AUC {readout.cv_score:.3f}); "
              f"ball drive = base {args.base:g} + urgency (size weight {args.size_weight:g}), "
              f"item drive = urgency (stake {args.item_stake:g}"
              f"{', diagonal' if args.item_diagonal else ''}), cap {args.cap:g}", flush=True)
    elif args.items:
        readout = train_items_readout(brain, base=args.base, cap=args.cap,
                                      size_weight=args.size_weight, item_stake=args.item_stake,
                                      diagonal=args.item_diagonal)
        print(f"decoder: trained readout on the shared chase channel (AUC {readout.cv_score:.3f}); "
              f"ball drive = base {args.base:g} + urgency (size weight {args.size_weight:g}), "
              f"item drive = urgency (stake {args.item_stake:g}"
              f"{', diagonal' if args.item_diagonal else ''}), cap {args.cap:g}", flush=True)
    else:
        readout = train_readout(brain)
        print(f"readout: cross-validated AUC {readout.cv_score:.3f} on a synthetic L/R sweep", flush=True)

    # In the default mode a decision costs 8 connectome steps (~50 ms), far more than
    # a 60 fps frame, so the brain runs in a thread at ~19 decisions/s and the main
    # loop reads its latest answer. In --continuous mode the brain keeps its state and
    # takes one step per frame (~4 ms, inside a frame), so it is stepped inline and no
    # thread is needed.
    box: dict = {"ball": None, "item": None, "want": None, "p": 0.5, "spikes": 0, "stop": False}
    lock = threading.Lock()

    def worker() -> None:
        while not box["stop"]:
            ball = box["ball"]
            if ball is None:
                time.sleep(0.002)
                continue
            dx, y, vy = ball
            if args.items:
                item = box["item"]
                db = args.base + object_demand(y, vy, dx, stake=STAKE_BALL, size_weight=args.size_weight)
                if item is None:
                    idx, di = None, 0.0
                else:
                    idx, iy, ivy = item
                    di = object_demand(iy, ivy, idx, stake=args.item_stake,
                                       diagonal=args.item_diagonal)
                p = float(readout.predict(dn_features_chase2(brain, dx, db, idx, di, cap=args.cap)))
            else:
                p = float(readout.predict(dn_features(brain, dx)))
            with lock:
                box["want"] = "right" if p >= 0.5 else "left"
                box["p"] = p
                box["spikes"] = int(brain.fired.size)

    if not args.continuous:
        brain_thread = threading.Thread(target=worker, daemon=True)
        brain_thread.start()

    a = RetroidAdapter(rom=args.rom, window="null" if args.headless else "SDL2",
                       scale=args.scale, sound_volume=args.volume)
    # PyBoy's own real-time throttle also sleeps, so it stacks with the per-frame cap
    # below and a "40 fps" run lands near 30. Hand the clock to the limiter alone so
    # `--fps` is the actual frame rate (and the fly gets the time it needs).
    a.pb.set_emulation_speed(0)
    if args.scene:
        a.load_scene(args.scene)
        a.step(6)   # prime the ball-motion history for `ball_live`
        print(f"scene: {args.scene}", flush=True)
    else:
        a.reset_to_play()
        a.launch()
    print("level 1 started. left/right are chosen by the connectome. Ctrl-C to stop.", flush=True)

    # Cap the loop at --fps. Pacing is *relative* (sleep the rest of this frame),
    # not an absolute deadline, so a slow frame never makes the loop burst at 60 to
    # "catch up": over --fps it just runs slower. The brain makes ~19 decisions/s,
    # so a lower cap keeps decisions closer to one per game frame.
    frame_dt = 1.0 / args.fps
    held = None
    prev_ball_y = prev_item_y = None
    dx_sum = dx_n = relaunches = losses = 0
    items_caught = items_missed = item_gaps = 0
    prev_item = None
    prev_live = True
    ball_gone = 0
    try:
        for step in range(args.steps):
            frame_start = time.perf_counter()
            st = a.state()
            ball_vy = 0.0 if (st.ball_y is None or prev_ball_y is None) else st.ball_y - prev_ball_y
            item_vy = 0.0 if (st.item_y is None or prev_item_y is None) else st.item_y - prev_item_y
            prev_ball_y, prev_item_y = st.ball_y, st.item_y
            live = a.ball_live()
            if prev_live and not live:
                losses += 1
            prev_live = live
            cur_item = None if st.item_x is None else (st.item_x, st.item_y)
            if prev_item is not None and cur_item is None:
                px, py = prev_item
                under = st.paddle_x is not None and abs(px - st.paddle_x) <= 12
                if py >= 128 and under:
                    items_caught += 1
                elif py >= 128:
                    items_missed += 1
                else:
                    item_gaps += 1
            prev_item = cur_item
            want, p, spikes = held, box["p"], box["spikes"]
            if not a.ball_live():
                ball_gone += 1
                box["ball"] = None
                if ball_gone == 1:
                    relaunches += 1
                if st.ball_x is not None:
                    # Ball resting on the paddle: hold A until it launches (a short
                    # pulse is not enough -- the game wants the button held).
                    want = "a"
                elif a.menu_text():
                    # GAME OVER or a menu: mash A so the fly moves on by itself. Pulse
                    # it instead of holding, since these screens want distinct presses.
                    want = "a" if (ball_gone // 2) % 2 == 0 else None
                elif ball_gone > args.exit_after:
                    # Gone far past a normal loss with no menu: the level is likely
                    # cleared and its exit is on the right. Keep asking for A too, since
                    # a continue screen whose letters the detector missed looks the same.
                    want = "a" if (ball_gone // 3) % 2 == 0 else "right"
                else:
                    # Ball gone (a loss animation): pulse A to serve as soon as it returns.
                    want = "a" if (ball_gone // 3) % 2 == 0 else None
                if want != held:
                    if held:
                        a.release(held)
                    if want:
                        a.hold(want)
                    held = want
                a.step(1)
            else:
                ball_gone = 0
                box["ball"] = None if st.dx is None else (st.dx, st.ball_y, ball_vy)
                box["item"] = None if (st.item_dx is None or st.item_y is None) else (st.item_dx, st.item_y, item_vy)
                if fly is not None:
                    db = args.base + object_demand(st.ball_y, ball_vy, st.dx,
                                                   stake=STAKE_BALL, size_weight=args.size_weight)
                    if args.items and st.item_dx is not None:
                        di = object_demand(st.item_y, item_vy, st.item_dx,
                                           stake=args.item_stake, diagonal=args.item_diagonal)
                        f = fly.step(st.dx, db, st.item_dx, di, cap=args.cap)
                    else:
                        f = fly.step(st.dx, db, None, None, cap=args.cap)
                    p = float(readout.predict(f))
                    want = "right" if p >= 0.5 else "left"
                    spikes = int(brain.fired.size)
                else:
                    with lock:
                        want, p, spikes = box["want"] or held, box["p"], box["spikes"]
                if want != held:
                    if held:
                        a.release(held)
                    a.hold(want)
                    held = want
                a.step(1)
            # Relative cap: never sleep on a stalled frame (no 60 fps catch-up burst),
            # and never exceed --fps. The relaunch branch paces too, so the press above
            # is the only burst left, and it happens with the ball already dead.
            if frame_dt:
                delay = frame_dt - (time.perf_counter() - frame_start)
                if delay > 0:
                    time.sleep(delay)
            if st.dx is not None:
                dx_sum += abs(st.dx)
                dx_n += 1
            if step % 30 == 0:
                item = box["item"]
                item_tag = "  -  " if item is None else "%+4.0f" % item[0]
                bx = "  -" if st.ball_x is None else "%3d" % st.ball_x
                px = "  -  " if st.paddle_x is None else "%5.1f" % st.paddle_x
                dxs = "  -  " if st.dx is None else "%+5.0f" % st.dx
                print(f"  step {step:4d}  ball x={bx}  paddle x={px}  dx={dxs}  item={item_tag}"
                      f"  p(right)={p:.2f} -> {want}  | DN spikes {spikes}", flush=True)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        box["stop"] = True
        if held:
            a.release(held)
        a.close()
        if dx_n:
            print("summary: %d frames in play, mean |dx| %.1f px, ball losses %d "
                  "(relaunch presses %d), items caught %d missed %d gap %d"
                  % (dx_n, dx_sum / dx_n, losses, relaunches, items_caught, items_missed, item_gaps),
                  flush=True)


if __name__ == "__main__":
    main()
