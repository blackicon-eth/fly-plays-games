# Retroid (Game Boy)

*Part of [fly-plays-games](../README.md): a real fruit-fly connectome driving games.*

**Status: planned.** The adapter and the numbers are not here yet; this file
records what we intend to do and why.

## The game

[Retroid](https://jonas-fischbach.itch.io/retroid) is a free, non-commercial
homebrew **Arkanoid / Breakout** clone for the original Game Boy, by **Jonas
Fischbach** (2016). A paddle at the bottom bounces a ball into a wall of bricks.

It is a better fit for the fly than it looks:

* **One axis, one decision.** The paddle only moves left/right, which is exactly
  the fly's left/right output. The natural control axis is horizontal, so no
  remapping is needed.
* **No memory.** The ball and the bricks are visible; nothing has to be
  remembered across steps. That removes the fly's main limitation.
* **A visible reflex loop.** "Keep the paddle under the ball" is continuous
  tracking, and it is forgiving about timing -- unlike a rhythm game, where the
  harness would need a precision the fly does not have.

The Game Boy ROM runs under [PyBoy](https://github.com/Baekalfen/PyBoy), the same
emulator used for the Pokémon Red chapter, so the pipeline carries over:

```
screen / object offset ─▶ visual projection neurons ─▶ the connectome ─▶ descending neurons ─▶ readout ─▶ paddle
```

## What is needed to make it real

1. **The ROM.** Free from the author's itch.io page. Like the Pokémon ROM, it will
   **not** be committed; the reader supplies it at `retroid/roms/retroid.gb`.
2. **An adapter.** Two ways to find the paddle and the ball:
   * read the game's **RAM** (needs the source or some reverse-engineering), or
   * read the **sprites in OAM** and recognise the paddle (a wide sprite near the
     bottom that moves horizontally) and the ball (a small sprite that bounces).
     This is more generic and would be reusable for other brick-breakers.
3. **The same readout.** Train a readout on the descending-neuron trace to output
   left/right, then drive the paddle. As everywhere in this project, the position
   of the ball is **handed to** the visual front end; the connectome only does the
   transform from visual projection neurons to descending neurons.
4. **An ablation.** Run the same game with the real connectome and with shuffled
   weights, and report the difference. That is the only honest way to say how much
   the fly contributes.

## Caveats we already expect

* **Ball speed.** If the ball is fast, "follow the current `x`" arrives late; the
  fly's loom/proximity signal (angular size) carries some time-to-contact, but our
  readout currently throws that dimension away.
* **Power-up capsules.** Catching a falling capsule is a **second decision**
  (chase the ball or the bonus), which leaves the pure reflex. Either a hand-coded
  priority rule (scaffolding) or ignore them.
* **Levels with enemies / multiple balls** are beyond a single reflex.

## Running it (once implemented)

```sh
python retroid/play_live.py --scene level1 --scale 4
```

The command above is aspirational; nothing under `retroid/` runs yet.
