# Retroid (Game Boy)

*Part of [fly-plays-games](../README.md): a real fruit-fly connectome driving games.*

**Status: working.** The fly connectome tracks the ball with the paddle. Same
brain, same visual front end and same readout as the Pokémon Red chapter; only
the game adapter is new.

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

The fly keeps the ball alive about as well as a scripted tracker that reads the
same sprite position:

| Driver | first ball lost | mean \|dx\| |
| --- | --- | --- |
| scripted tracker (dead-zone 2 px) | frame 2355 | 5.2 px |
| **fly (connectome + readout)** | frame 3549 | 5.1 px |

## What the ablation says

The honest test is not "can the fly play" but "does the connectome matter". The
`record_fly.py --ablate` option scrambles the connectome's weights in place and
**retrains the readout** on the scrambled brain:

| Ablation | frames in play (of 800) | mean \|dx\| |
| --- | --- | --- |
| `none` (real connectome) | 796 | 4.6 px |
| `shuffle` (weights shuffled) | 796 | 4.6 px |
| `silence` (weights zeroed) | 676 | 81 px |

The deflating but honest result: **a shuffled connectome tracks the ball exactly
as well as the real one**, because the task is one-dimensional and the side is
linearly decodable from almost any encoding. Zeroing the weights breaks it
entirely. So the connectome is genuinely in the loop and genuinely produces the
button presses, but for a single left/right reflex it is **not necessary**. It
would take a task whose structure depends on the specific wiring to change that.

(Note: the readout's cross-validated AUC reads 1.000 even for `silence`, where the
descending-neuron trace is identical for left and right. For degenerate features
the AUC is not meaningful, which is why the table above reports the closed-loop
score instead.)

## Running it

The ROM is free from the author's itch.io page and is **not committed**. Put it at
`retroid/roms/Retroid.gb` (any filename works via `--rom`).

```sh
python retroid/play_live.py --scale 4                  # watch it, with a window
python retroid/render/record_fly.py --steps 1200       # record a run to render/fly_drive.npz
python retroid/render/render_fly.py                    # render that npz to media/retroid_fly.mp4
python retroid/render/record_fly.py --ablate shuffle   # the control condition
```

## Caveats

* **Ball speed.** The ball moves up to 3 px/frame and the paddle 2 px/frame, so
  the tracking is a real control problem, not a formality. The fly holds up here.
* **Power-ups.** A falling capsule is a *second* decision (chase the ball or the
  bonus) and is outside a single reflex; the current runs ignore them.
* **Levels with enemies or multiple balls** are out of reach.
* **Sprite identification** assumes the ball stays tile `$00`; a level that
  reuses that tile for something else would confuse the adapter.
