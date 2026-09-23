"""Record a continuous-mode run (level 1, the stages, or the boss) to the npz
render_fly.py reads.

This is `play_live.py --continuous` without a window: one connectome step per game
frame, the same drives (the ball's urgency, the falling item's, and on the boss the
escape neurons DNp01 driven by the shots' looming). It exists because
`record_fly.py` still uses the older threaded readout, while the boss needs the
continuous one plus the escape.

    python retroid/render/record_run.py --scene level1 --stop-at-loss --out render/level1.npz
    python retroid/render/record_run.py --scene boss --stage 21 --escape --stop-at-clear --out render/boss.npz
    python retroid/render/record_run.py --scene level1 --stop-at-defeat --out render/stages.npz
    python retroid/render/record_run.py --scene level1 --items --stop-at-defeat --out render/stages_items.npz

Every game frame is recorded, including the lost balls and the relaunch. The run
stops at the fly's first ball loss (`--stop-at-loss`), when the paddle leaves the
right edge after a cleared level (`--stop-at-clear`), or at the game-over screen
after the last ball at x00 (`--stop-at-defeat`). With no stop flag it plays to
`--steps`.
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

GAME = Path(__file__).resolve().parents[1]      # retroid/
ROOT = GAME.parent                              # repository root (data/, fly-ai/)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fly-ai"))
sys.path.insert(0, str(GAME))

from flybrain import FlyBrain
from retroidsim import (DEFAULT_ROM, STAKE_BALL, STAKE_ESCAPE, ContinuousChase,
                        RetroidAdapter, escape_sides, object_demand,
                        train_continuous_readout)
from retroidsim.adapter import STAGE_ADDR

DIRIDX = {"up": 0, "down": 1, "left": 2, "right": 3}
BTNIDX = {"up": 0, "down": 1, "left": 2, "right": 3, "a": 4, "b": 5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=8000, help="game frames to drive")
    ap.add_argument("--rom", default=str(DEFAULT_ROM))
    ap.add_argument("--window", default="null")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--data", default=str(ROOT / "data" / "fly-data"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scene", default="level1", help="starting scene name (level1 or boss)")
    ap.add_argument("--stage", type=int, default=None, help="freeze the stage number (21 = boss)")
    ap.add_argument("--escape", action="store_true", help="drive the escape neurons (DNp01) with the shots' looming")
    ap.add_argument("--escape-stake", type=float, default=STAKE_ESCAPE)
    ap.add_argument("--escape-gain", type=float, default=0.08)
    ap.add_argument("--escape-ball-safe", type=float, default=0.8)
    ap.add_argument("--base", type=float, default=0.8)
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--size-weight", type=float, default=0.1)
    ap.add_argument("--items", action="store_true",
                    help="also inject the falling item and chase it when the ball is safe")
    ap.add_argument("--item-stake", type=float, default=0.8,
                    help="how much the item's demand counts (bigger = the item wins from higher up)")
    ap.add_argument("--item-diagonal", action="store_true",
                    help="item distance = straight line paddle-item, so a far one looms less")
    ap.add_argument("--stop-at-loss", action="store_true", help="stop at the fly's first ball loss")
    ap.add_argument("--stop-at-clear", action="store_true", help="stop when the paddle leaves the right edge")
    ap.add_argument("--stop-at-defeat", action="store_true",
                    help="stop at the game-over screen, after the last ball at x00")
    ap.add_argument("--exit-after", type=int, default=24,
                    help="dead frames before driving right to a cleared level's exit")
    ap.add_argument("--tail-seconds", type=float, default=4.0,
                    help="extra seconds recorded after the stop, so the win is visible")
    ap.add_argument("--audio-out", default="", help="write the game audio to this WAV (default: <out>.wav)")
    ap.add_argument("--out", default=str(GAME / "render" / "run.npz"))
    args = ap.parse_args()

    print("loading the connectome...", flush=True)
    brain = FlyBrain(data=args.data, device=args.device)
    readout = train_continuous_readout(brain, base=args.base, cap=args.cap,
                                       size_weight=args.size_weight, item_stake=args.item_stake,
                                       diagonal=args.item_diagonal, with_item=True)
    fly = ContinuousChase(brain)
    print("decoder: continuous brain, one step per frame (AUC %.3f)" % readout.cv_score, flush=True)

    a = RetroidAdapter(rom=args.rom, window=args.window, scale=args.scale, stage=args.stage)
    a.reset_to_play(name=args.scene, stage=args.stage)
    a.launch()

    prev_projs: list = []
    prev_ball_x = prev_ball_y = None
    prev_item_y = None
    FR, F, DIRS, BTN, TRACK = [], [], [], [], []
    AUDIO: list = []
    SHOTS, LIVES = [], []
    held = None
    losses = 0
    ball_gone = 0

    def rec(name, st):
        FR.append(a.screen(gray=False))
        F.append(brain.fired.astype(np.int32))
        DIRS.append(DIRIDX.get(name, 3))
        BTN.append(BTNIDX.get(name, 6))
        row = [st.ball_x if st.ball_x is not None else -1,
               st.ball_y if st.ball_y is not None else -1,
               st.paddle_x if st.paddle_x is not None else -1]
        if args.items:
            row += [st.item_x if st.item_x is not None else -1,
                    st.item_y if st.item_y is not None else -1]
        TRACK.append(tuple(row))
        AUDIO.append(a.pb.sound.ndarray.copy())
        SHOTS.append(len(a.projectiles()))
        lv = a.lives()
        LIVES.append(-1 if lv is None else lv)

    def set_held(name):
        nonlocal held
        if name != held:
            if held is not None:
                a.release(held)
            if name is not None:
                a.hold(name)
            held = name

    stop = "steps"
    last_live_stage = int(a.pb.memory[STAGE_ADDR])
    dead_stage = last_live_stage
    for step in range(args.steps):
        st = a.state()
        projs = a.projectiles()
        lv = a.lives()
        stage = int(a.pb.memory[STAGE_ADDR])
        pos_ok = prev_ball_x is not None and prev_ball_y is not None
        ball_vy = 0.0 if (st.ball_y is None or not pos_ok) else st.ball_y - prev_ball_y
        ball_vx = 0.0 if (st.ball_x is None or not pos_ok) else st.ball_x - prev_ball_x
        item_vy = 0.0 if (st.item_y is None or prev_item_y is None) else st.item_y - prev_item_y
        prev_ball_x, prev_ball_y = st.ball_x, st.ball_y
        prev_item_y = st.item_y
        # At x00 the game still serves one last ball and the fly plays it; the run
        # only ends when that ball is lost and the game-over menu appears. Stop
        # before the dead-ball scaffolding mashes A through it and restarts.
        if args.stop_at_defeat and lv == 0 and a.menu_text():
            stop = "defeat (game over) at step %d" % step
            break
        if a.ball_live():
            ball_gone = 0
            last_live_stage = stage
            db = args.base + object_demand(st.ball_y, ball_vy, st.dx,
                                           stake=STAKE_BALL, size_weight=args.size_weight)
            esc_l = esc_r = 0.0
            if (args.escape and projs and st.paddle_x is not None
                    and object_demand(st.ball_y, ball_vy, st.dx, stake=STAKE_BALL,
                                      size_weight=args.size_weight) <= args.escape_ball_safe):
                esc_l, esc_r = escape_sides(projs, st.paddle_x, prev_projs, stake=args.escape_stake)
            if args.items and st.item_dx is not None:
                di = object_demand(st.item_y, item_vy, st.item_dx,
                                   stake=args.item_stake, diagonal=args.item_diagonal)
                f = fly.step(st.dx, db, st.item_dx, di, cap=args.cap, vx_a=ball_vx,
                             escape_l=esc_l, escape_r=esc_r)
            else:
                f = fly.step(st.dx, db, None, None, cap=args.cap, vx_a=ball_vx,
                             escape_l=esc_l, escape_r=esc_r)
            p = float(readout.predict(f))
            if args.escape:
                turn = (p - 0.5) - args.escape_gain * fly.escape_level()
                want = "right" if turn >= 0.0 else "left"
            else:
                want = "right" if p >= 0.5 else "left"
            set_held(want)
            rec(want, st)
            a.step(1)
            if step % 100 == 0:
                print("step %4d  stage %2d  lives %s  dx=%+5.0f  p=%.2f -> %s"
                      % (step, int(a.pb.memory[STAGE_ADDR]), lv, st.dx, p, want), flush=True)
            if args.stop_at_clear and st.paddle_x is not None and st.paddle_x > 158:
                stop = "cleared at step %d" % step
                break
        else:
            ball_gone += 1
            if ball_gone == 1:
                dead_stage = last_live_stage
                losses += 1
                print("step %4d  ball lost (lives %s); relaunching" % (step, lv), flush=True)
            # The dead-ball scaffolding is play_live.py's: hold A to serve, mash
            # through a menu, and drive right to a cleared level's exit so the run
            # moves on to the next stage. Every frame is recorded. Once the stage
            # number changes the exit worked, so stop driving right and serve.
            if stage != dead_stage:
                want = "a" if (ball_gone // 3) % 2 == 0 else None
            elif st.ball_x is not None:
                want = "a"
            elif a.menu_text():
                want = "a" if (ball_gone // 2) % 2 == 0 else None
            elif ball_gone > args.exit_after:
                want = "a" if (ball_gone // 3) % 2 == 0 else "right"
            else:
                want = "a" if (ball_gone // 3) % 2 == 0 else None
            set_held(want)
            rec(want, st)
            a.step(1)
            if args.stop_at_loss and losses == 1 and ball_gone == 1:
                stop = "ball lost at step %d" % step
                break
            if args.stop_at_clear and ball_gone > 90:
                stop = "level cleared (ball stayed gone) at step %d" % step
                break
        prev_projs = projs

    set_held(None)
    # A few extra seconds after the stop, so a win is not cut off at the final hit.
    for _ in range(int(args.tail_seconds * 60)):
        rec(None, a.state())
        a.step(1)
    rate = a.pb.sound.sample_rate
    a.close()
    lens = np.array([len(x) for x in F])
    np.savez(args.out, frames=np.array(FR, np.uint8), fired=np.concatenate(F),
             starts=np.concatenate([[0], np.cumsum(lens)]), dirs=np.array(DIRS, np.int8),
             btns=np.array(BTN, np.int8), track=np.array(TRACK, np.int16),
             shots=np.array(SHOTS, np.int8), lives=np.array(LIVES, np.int8),
             ablate=np.array("none"))
    print("saved", args.out, "frames", len(FR), "| stopped:", stop, flush=True)
    audio_out = args.audio_out or str(Path(args.out).with_suffix(".wav"))
    if AUDIO:
        stereo = np.concatenate(AUDIO, axis=0).astype(np.int16) << 8
        with wave.open(audio_out, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(stereo.tobytes())
        print("saved", audio_out, "samples", len(stereo), flush=True)


if __name__ == "__main__":
    main()
