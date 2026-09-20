"""How long does the fly keep the ball alive? The game's own outcome, not a proxy.

Every other Retroid script scores a *proxy* (mean |dx|, or which signal won a
conflict). Retroid has a real outcome: when the ball passes the paddle you lose a
life, and the ball sprite disappears from the OAM table. So the honest score is
frames survived before the first lost ball, capped at `--budget` (a capped run
means the policy never lost the ball).

The loop is the same as `record_fly.py` (offset -> visual channels -> connectome
-> readout -> left/right), plus controls: `oracle` (a perfect tracker, the
ceiling), `random`, `still`, and the ablations `shuffle`/`rewire`/`silence`. The
ball's position is handed to the visual front end, as everywhere; the readout is
retrained after each ablation. `shuffle`/`rewire`/`random` are run over several
seeds because their outcome is stochastic; `none`/`silence`/`oracle`/`still` are
deterministic from a fixed scene and run once.

    python retroid/render/survival.py --scenes level1 --budget 2000
    python retroid/render/survival.py --scenes level1 level1_b --policies none shuffle rewire

Writes an npz of per-run results and prints a table (mean +- std over seeds).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

GAME = Path(__file__).resolve().parents[1]      # retroid/
ROOT = GAME.parent                              # repository root (data/, fly-ai/)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fly-ai"))
sys.path.insert(0, str(GAME))

from flybrain import FlyBrain
from retroidsim import DEFAULT_ROM, RetroidAdapter, ablate, dn_features, train_readout

FLY_POLICIES = ("none", "shuffle", "rewire", "silence")
ALL_POLICIES = FLY_POLICIES + ("oracle", "random", "still")
SEEDED = ("shuffle", "rewire", "random")   # policies whose result changes with the seed
LAUNCH_HOLD = 10


def wait_for_ball(a: RetroidAdapter, timeout: int = 900) -> bool:
    """Step until the ball sprite appears (covers the paddle slide-in after a loss)."""
    for _ in range(timeout):
        if a.ball() is not None:
            return True
        a.step(1)
    return False


def prepare(a: RetroidAdapter, probe: int = 8) -> bool:
    """Get a loaded scene to a launched, in-play ball.

    A scene may already be mid-flight (nothing to do) or have the ball resting on
    the paddle (launch it). Resting and mid-flight are told apart by motion, not
    height, because a caught ball dips to y=131 at the bottom of a bounce.
    """
    if not wait_for_ball(a):
        return False
    for _ in range(probe):
        a.step(1)
        if a.ball_live():
            return True
    a.press("a", LAUNCH_HOLD)
    for _ in range(90):
        a.step(1)
        if a.ball_live():
            return True
    return False


def make_policy(name: str, readout, decide, rng):
    """Return `act(state) -> button or None` for one policy."""
    if name in FLY_POLICIES:
        def act(st):
            if st.dx is None:
                return None
            p = float(readout.predict(dn_features(decide, st.dx)))
            return "right" if p >= 0.5 else "left"
        return act
    if name == "oracle":
        def act(st):
            dx = st.dx
            if dx is None:
                return None
            if dx > 2.0:
                return "right"
            if dx < -2.0:
                return "left"
            return None
        return act
    if name == "still":
        return lambda st: None
    if name == "random":
        box = {"dir": None, "left": 0}

        def act(st):
            if box["left"] <= 0:
                box["dir"] = "right" if rng.random() < 0.5 else "left"
                box["left"] = 4
            box["left"] -= 1
            return box["dir"]
        return act
    raise ValueError(f"unknown policy: {name}")


def run_episode(a: RetroidAdapter, act, budget: int) -> tuple[int, bool]:
    """Drive until the ball is lost (returns frames survived, lost?)."""
    held, frames = None, 0
    for _ in range(budget):
        if a.ball() is None:
            break
        st = a.state()
        want = act(st)
        if want != held:
            if held:
                a.release(held)
            if want:
                a.hold(want)
            held = want
        a.step(1)
        frames += 1
    if held:
        a.release(held)
    return frames, a.ball() is None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=None, help="scene names (default: level1)")
    ap.add_argument("--policies", nargs="*", default=list(ALL_POLICIES), choices=ALL_POLICIES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=2000, help="cap per episode (game frames)")
    ap.add_argument("--rom", default=str(DEFAULT_ROM))
    ap.add_argument("--data", default=str(ROOT / "data" / "fly-data"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(GAME / "render" / "survival.npz"))
    args = ap.parse_args()

    scenes = args.scenes or ["level1"]
    a = RetroidAdapter(rom=args.rom)
    for s in scenes:
        if not a.has_scene(s):
            if s == "level1":
                print("building the level1 scene...", flush=True)
                a.reset_to_play(force=True)
            else:
                raise SystemExit(f"no scene {s!r}; make it first (see make_scenes.py)")

    results: dict[tuple[str, int], list] = {}
    t0 = time.time()

    def eval_config(policy: str, seed: int, readout, decide) -> None:
        rng = np.random.default_rng(1000 + seed)
        act = make_policy(policy, readout, decide, rng)
        rows = []
        for s in scenes:
            a.load_scene(s)
            if not prepare(a):
                print(f"  {policy:8s} seed{seed} {s}: could not start a ball", flush=True)
                continue
            frames, lost = run_episode(a, act, args.budget)
            rows.append((s, frames, lost))
            print(f"  {policy:8s} seed{seed} {s:8s}: survived {frames:5d} frames"
                  f"{'' if lost else '  (capped)'}   [{time.time()-t0:5.0f}s]", flush=True)
        results[(policy, seed)] = rows

    for policy in [p for p in args.policies if p in FLY_POLICIES]:
        seeds = args.seeds if policy in SEEDED else [0]
        for seed in seeds:
            decide = FlyBrain(data=args.data, device=args.device)
            ablate(decide, policy, seed)
            readout = train_readout(decide)
            print(f"policy={policy} seed={seed}  readout AUC {readout.cv_score:.3f}", flush=True)
            eval_config(policy, seed, readout, decide)
            del decide

    for policy in [p for p in args.policies if p not in FLY_POLICIES]:
        seeds = args.seeds if policy in SEEDED else [0]
        for seed in seeds:
            eval_config(policy, seed, None, None)

    a.close()
    flat = [(p, s, sc, f, int(l)) for (p, s), rows in results.items() for (sc, f, l) in rows]
    np.savez(args.out, results=np.array(flat, dtype=object))

    print(f"\n== survival (frames to first lost ball, cap {args.budget}) ==", flush=True)
    print("%-8s %5s  %-22s %s" % ("policy", "seed", "mean +- std", "per scene"), flush=True)
    for (policy, seed), rows in sorted(results.items()):
        if not rows:
            continue
        fr = np.array([r[1] for r in rows], float)
        per = " ".join(f"{r[0]}={r[1]}{'' if r[2] else '*'}" for r in rows)
        print("%-8s %5d  %8.1f +- %-10.1f %s" % (policy, seed, fr.mean(), fr.std(), per), flush=True)
    print("(* = capped: the ball was never lost)", flush=True)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
