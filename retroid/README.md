# Retroid (Game Boy)

*Part of [fly-plays-games](../README.md): a real fruit-fly connectome driving games.*

**Status: working (level 1).** The fly connectome tracks the ball with the paddle
and keeps it alive; an ablation shows the specific *wiring* -- not the weight
values -- is what carries the signal. Same brain, same visual front end and same
readout as the Pokémon Red chapter; only the game adapter is new.

![The fly connectome driving Retroid: the fly on a gamepad, the game with a ball/paddle track, and the connectome's activity](media/retroid_fly.gif)

## The game

[Retroid](https://jonas-fischbach.itch.io/retroid) is a free, non-commercial
homebrew **Arkanoid / Breakout** clone for the original Game Boy, by **Jonas
Fischbach** (2016). A paddle at the bottom bounces a ball into a wall of bricks.

It fits the fly far better than Pokémon does:

* **One axis, one decision.** The paddle only moves left/right, which is exactly
  the fly's left/right output. No remapping is needed.
* **No memory.** The ball and bricks are on screen; nothing has to be carried
  across steps. The fly's missing long-term memory does not matter here.
* **A pure reflex.** "Keep the paddle under the ball" is continuous tracking and
  is forgiving about timing.

## How it works

The Game Boy draws the ball and paddle as hardware sprites, so the adapter reads
them from the OAM table at `$FE00` instead of reverse-engineering WRAM. Each of
the 40 slots is four bytes `(screen y, screen x, tile, attributes)`; Retroid keeps
the **ball in tile `$00`** and the **three-tile paddle in tiles `$01`–`$03`**. The
adapter finds them by tile, so it does not care about the slot order. Everything
else is the shared pipeline:

```
ball offset dx ─▶ visual projection neurons ─▶ the connectome ─▶ descending neurons ─▶ readout ─▶ left/right
                 (LPLC2 / LC4 / LPLC1 / LC10a)   (166,700, frozen)   (~1,300)          (trained)
```

* **Adapter** (`retroidsim/adapter.py`). Runs the ROM under PyBoy, navigates the
  fixed opening (the paddle slides in from the left and ignores input until it
  stops), bookmarks the level-1 start as a scene, launches the ball, and exposes
  `paddle_x`, `ball_x/y` and `dx = ball_x - paddle_x`.
* **Encoder** (`retroidsim/encoder.py`). Injects `dx` into the fly's visual
  projection neurons through `FeatureDetectors`, then reads the descending-neuron
  trace. As in Pokémon, the ball's position is **handed to** the visual front end
  (the photoreceptor route fades at the lamina in a spiking model); the connectome
  only does visual projection neurons → descending neurons.
* **Readout**. A logistic map over the descending-neuron trace, fit at startup on
  a synthetic sweep of offsets labelled by side (`dx > 0`). The connectome's
  weights never change; only this readout is trained.

## Results

From the level-1 start, the fly keeps the ball alive longer than a scripted
dead-zone tracker that reads the same sprite position:

| Driver | first ball lost |
| --- | --- |
| scripted tracker (dead-zone 2 px) | frame 2344 |
| **fly (connectome + readout)** | frame 4002 |

## What the ablation says

The honest test is not "can the fly play" but "does the connectome matter". The
`record_fly.py --ablate` option scrambles the connectome's weights in place and
**retrains the readout** on the scrambled brain:

| Ablation | frames in play (of 800) | mean \|dx\| |
| --- | --- | --- |
| `none` (real connectome) | 800 | 2.9 px |
| `shuffle` (weights shuffled) | 800 | 2.9 px |
| `rewire` (topology randomized) | 749 | 30.6 px |
| `silence` (weights zeroed) | 749 | 30.6 px |

The deflating but honest result: **a shuffled connectome tracks the ball exactly
as well as the real one**, because the task is one-dimensional and the side is
linearly decodable from almost any encoding. So the connectome is genuinely in the
loop and genuinely produces the button presses, but for a single left/right reflex
its **weight values are not necessary**. What *is* necessary is the wiring: break
the topology (`rewire`) or the synapses (`silence`) and the reflex collapses.

(Note: the readout's cross-validated AUC reads 1.000 even for `silence`, where the
descending-neuron trace is identical for left and right. For degenerate features
the AUC is not meaningful, which is why the table above reports the closed-loop
score instead.)

Two ablations do break the reflex: **`rewire`** (every neuron keeps its number of
inputs but draws them from random neurons) and **`silence`**. Both flatten the
descending-neuron trace -- it becomes *identical* for every ball offset -- so the
readout outputs a constant and the paddle stops tracking. The side is therefore
carried by the **wiring**, not by the weight values. (The readout still *reports*
AUC 1.000 in both cases: the upstream `auc` scores tied predictions as if they were
ordered, so a constant output comes out at 1.000 instead of 0.5. Another reason the
closed-loop score above is the honest one.)

## Survival: the game's own outcome

`mean |dx|` is a proxy. Retroid has a real outcome: when the ball passes the
paddle you lose a life, and the ball sprite disappears from the OAM table. So the
honest score is **frames until the first lost ball**, with the fly relaunching
after each loss like a player (`render/survival.py`). Four starting scenes (the
level start plus three mid-flight snapshots from `render/make_scenes.py`), cap 1500
frames; `shuffle`/`rewire`/`random` run over two seeds.

| Driver | frames to first lost ball (mean over 4 scenes) |
| --- | --- |
| **fly, real connectome** | **>1500 (never lost)** |
| fly, weights shuffled | >1500 (never lost) |
| fly, topology rewired | 204 |
| fly, weights silenced | 204 |
| scripted tracker (oracle) | 1372 |
| random paddle | 111 / 174 |
| no input at all | 162 |

The fly keeps the ball alive through all 1500 frames from every scene, and so does
a fly whose *weights* are scrambled. But randomise the *wiring* -- or zero the
weights -- and it loses the ball in about 200 frames, no better than a random
paddle. `rewire` and `silence` are numerically identical here because both flatten
the descending-neuron trace to a constant, so the paddle gets the same fixed
command. This is the same picture as the proxy, now with a real consequence: for a
single reflex the weight values are not needed, but the specific wiring is.

(Without the cap, the fly loses its first ball at frame 4002 from the level start,
against 2344 for the scripted tracker.)

## Two signals: the ball against a falling item

The single-reflex tests above show the connectome's *weights* are not necessary
for a one-dimensional task. So here is a conflict. The ball drives the chase channel
(`opp` → LC10a), a falling item drives the projectile channel (`shots` → LPLC1),
and both are injected at once, on opposite sides. The readout is trained only on
**single-object** trials (ball alone, item alone), so it never sees a conflict:
whatever it does when both are present is the connectome arbitrating.

`render/conflict.py` sweeps the item's angular size (its urgency):

![Two-signal conflict: P(right) as the item grows, for the real connectome and with shuffled weights](media/conflict.png)

* **Real connectome.** Small items lose and the ball is followed. Past a size of
  about 10 the item wins and the fly abandons the ball. The effect is **lateral**:
  it only happens when the ball is on the right; with the ball on the left the
  item never wins.
* **Weights shuffled.** The item never wins at any size; the fly stays on the ball.
  The side is still decodable, so the network still works, just with a different
  arbitration.
* **Topology rewired.** Every neuron keeps the same number of incoming connections,
  but they now come from random neurons. The readout can no longer tell left from
  right **at all** (the output is a flat 0.48). So the left/right information lives
  in the **wiring**, while the arbitration is shaped by the weights.

So the wiring *does* matter. Rewiring the graph destroys the signal entirely, and
even scrambling only the weights changes the arbitration and its left/right
asymmetry. Together with the survival test, this is where damaging the connectome
changes the outcome.

The same thing happens in the game loop (`render/record_conflict.py`, the ball into
`opp`, the item into `shots`, readout trained on single objects only). When the ball
drifts far and an item falls on the other side, the fly abandons the ball for the
item, and catches it:

![The fly abandoning the ball for a falling item](media/retroid_conflict.gif)

In one 3000-frame run it followed the item on 96 conflict frames and the ball on 65.
But two separate runs diverge, so `render/conflict_paired.py` evaluates both
readouts on the **same** (ball dx, item dx): on 161 conflict frames the real
connectome follows the item **60%** of the time and the shuffled one **16%**, and
they disagree on **44%** of them. The item wins more in-game than in the size sweep
above because it is injected *growing* as it falls, which carries it past the size
where it out-competes the ball. The choices are always bang-bang frame to frame, so
this is a statistic, not a sustained decision. Caveats: this is a probe and a demo,
not a benchmark; the numbers depend on the encoder's growth rate and on the readout's
training.

## Running it

The ROM is free from the author's itch.io page and is **not committed**. Put it at
`retroid/roms/Retroid.gb` (any filename works via `--rom`).

```sh
python retroid/play_live.py --scale 4                  # watch it, with a window
python retroid/render/record_fly.py --steps 1200       # record a run to render/fly_drive.npz
python retroid/render/render_fly.py                    # render that npz to media/retroid_fly.mp4
python retroid/render/record_fly.py --ablate shuffle   # the control condition
python retroid/render/make_scenes.py --every 450       # extra mid-flight scenes
python retroid/render/survival.py --scenes level1 --budget 1500   # lives: the real outcome
python retroid/render/conflict.py --ablate none         # two-signal arbitration
python retroid/render/conflict.py --ablate shuffle      # weights scrambled
python retroid/render/conflict.py --ablate rewire       # topology scrambled
python retroid/render/record_conflict.py --steps 3000   # the conflict in the game loop
python retroid/render/conflict_paired.py --steps 2500   # both readouts on the same states
python retroid/render/render_fly.py --input render/conflict_play.npz --start 790 --end 1030
```

## Caveats

* **Ball speed.** The ball moves up to 3 px/frame and the paddle 2 px/frame, so
  the tracking is a real control problem, not a formality. The fly holds up here.
* **Power-ups.** A falling capsule is a *second* decision (chase the ball or the
  bonus) and is outside a single reflex; the current runs ignore them.
* **Levels with enemies or multiple balls** are out of reach.
* **Sprite identification** assumes the ball stays tile `$00`; a level that
  reuses that tile for something else would confuse the adapter.
* **Lost-ball detection.** A lost ball is the sprite disappearing from the OAM
  table, not the ball's height: a *caught* ball dips to y=131 at the bottom of a
  bounce, so a height threshold would misread every bounce as a lost ball and
  freeze the paddle for a few frames. (An earlier version did exactly that; the
  numbers above are from the fixed detection.)
