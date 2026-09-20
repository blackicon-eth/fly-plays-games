"""Run Retroid (a Game Boy Arkanoid) under PyBoy and expose it as a game object.

This is the only module that knows about PyBoy, the ROM and the sprite table.
Everything downstream (the encoder, the fly brain) talks to `RetroidAdapter` in
plain Python.

Retroid draws the paddle and the ball as hardware sprites, so we read them from
the OAM table at $FE00 instead of guessing at WRAM addresses: each of the 40
slots is four bytes (screen y, screen x, tile, attributes), and the game keeps
the ball in tile $00 and the three-tile paddle in tiles $01-$03. The adapter
identifies them by tile, so it does not depend on the slot order.

The opening is a fixed script that ignores input while the paddle slides in from
the left, so `reset_to_play` mashes A to reach the level, waits for the paddle to
settle at the centre, and bookmarks that moment once. Later resets are instant.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyboy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROM = PROJECT_ROOT / "roms" / "Retroid.gb"
SCENES_DIR = PROJECT_ROOT / "roms" / "scenes"

OAM_BASE = 0xFE00
OAM_SLOTS = 40
SCREEN_W, SCREEN_H = 160, 144

BALL_TILE = 0x00
PADDLE_TILES = (0x01, 0x02, 0x03)
ITEM_TILES = (0x18, 0x19)   # the falling capsule in level 1 (two animation frames)

BUTTONS = ("a", "b", "start", "select", "up", "down", "left", "right")

# Opening script (frames). Input is ignored until the paddle stops sliding in,
# so we only need to reach the level; `READY_DELAY` covers the slide and pause.
INTRO_TICKS = 200
INTRO_START_HOLD = 6
INTRO_A_PERIOD = 40
INTRO_A_HOLD = 6
READY_DELAY = 400
LAUNCH_HOLD = 10


@dataclass(frozen=True)
class RetroidState:
    """One moment of the game, as the fly's encoder sees it.

    `paddle_x` is the centre of the paddle in screen pixels (the paddle is three
    tiles wide); `ball_x`/`ball_y` are the ball's top-left corner. `dx` is the
    horizontal offset the fly reacts to.
    """

    paddle_x: float | None
    ball_x: int | None
    ball_y: int | None
    item_x: int | None = None
    item_y: int | None = None

    @property
    def dx(self) -> float | None:
        if self.paddle_x is None or self.ball_x is None:
            return None
        return float(self.ball_x - self.paddle_x)

    @property
    def item_dx(self) -> float | None:
        if self.paddle_x is None or self.item_x is None:
            return None
        return float(self.item_x - self.paddle_x)

    @property
    def in_play(self) -> bool:
        return self.ball_x is not None and self.paddle_x is not None


class RetroidAdapter:
    """A headless Retroid you can step, hold buttons on and read sprites from.

    `reset_to_play` returns the game to the level 1 start (ball on the paddle,
    paddle centred); `save_scene`/`load_scene` bookmark any other moment so
    experiments restart from the exact same spot. Scenes live in `roms/scenes/`
    next to the ROM and are not committed.
    """

    def __init__(self, rom: str | Path = DEFAULT_ROM, scenes_dir: str | Path | None = None,
                 window: str = "null", scale: int = 3):
        self.rom = Path(rom)
        if not self.rom.exists():
            raise FileNotFoundError(f"no ROM at {self.rom}; see README.md")
        self.scenes_dir = Path(scenes_dir) if scenes_dir else SCENES_DIR
        self.pb = pyboy.PyBoy(str(self.rom), window=window, scale=scale)
        self._ball_hist: deque = deque(maxlen=8)   # recent ball positions, for `ball_live`

    # ---- sprites --------------------------------------------------------------------

    def oam(self) -> np.ndarray:
        """The 40 hardware sprites as (y, x, tile, attributes), raw OAM bytes."""
        raw = np.asarray(self.pb.memory[OAM_BASE:OAM_BASE + OAM_SLOTS * 4], dtype=np.uint8)
        return raw.reshape(OAM_SLOTS, 4)

    def _visible(self) -> list[tuple[int, int, int]]:
        """Visible sprites as (screen x, screen y, tile). OAM y=0 means hidden,
        and both coordinates are biased by 8/16 for the 8x8 sprite."""
        out = []
        for y, x, tile, _ in self.oam():
            if y == 0:
                continue
            out.append((int(x) - 8, int(y) - 16, int(tile)))
        return out

    def ball(self) -> tuple[int, int] | None:
        for x, y, tile in self._visible():
            if tile == BALL_TILE:
                return x, y
        return None

    def item(self) -> tuple[int, int] | None:
        """The falling capsule, if one is on screen."""
        for x, y, tile in self._visible():
            if tile in ITEM_TILES:
                return x, y
        return None

    def paddle_x(self) -> float | None:
        xs = [x for x, _, tile in self._visible() if tile in PADDLE_TILES]
        if not xs:
            return None
        return (min(xs) + max(xs)) / 2 + 4

    def ball_live(self, window: int = 4) -> bool:
        """True while a launched ball is in play.

        A ball resting on the paddle, one that has just appeared, and a lost ball
        are all *not* live; a launched ball is. The test is motion over the last
        `window` frames, not the ball's height: a caught ball dips to y=131 at the
        bottom of a bounce, which a y-threshold would misread as "ball lost" and
        would freeze the paddle for a few frames at every bounce.
        """
        hist = list(self._ball_hist)[-(window + 1):]
        if len(hist) < window + 1 or any(b is None for b in hist):
            return False
        return any(hist[i] != hist[i - 1] for i in range(1, len(hist)))

    # ---- stepping and input ---------------------------------------------------------

    def step(self, frames: int = 1) -> None:
        for _ in range(frames):
            self.pb.tick(1)
            self._ball_hist.append(self.ball())

    def hold(self, button: str) -> None:
        self.pb.button_press(self._check(button))

    def release(self, button: str) -> None:
        self.pb.button_release(self._check(button))

    def press(self, button: str, frames: int = 6) -> None:
        """Hold `button` for `frames` frames, then release."""
        self.hold(button)
        self.step(frames)
        self.release(button)

    @staticmethod
    def _check(button: str) -> str:
        name = button.lower()
        if name not in BUTTONS:
            raise ValueError(f"button must be one of {BUTTONS}, not {button!r}")
        return name

    # ---- observation ----------------------------------------------------------------

    def screen(self, gray: bool = False) -> np.ndarray:
        """The 144x160 screen as RGB uint8, or luminance uint8 if `gray`."""
        rgba = np.asarray(self.pb.screen.ndarray)
        rgb = rgba[..., :3]
        if not gray:
            return rgb.copy()
        return np.asarray(rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32), np.uint8)

    def state(self) -> RetroidState:
        ball = self.ball()
        item = self.item()
        return RetroidState(
            paddle_x=self.paddle_x(),
            ball_x=ball[0] if ball else None,
            ball_y=ball[1] if ball else None,
            item_x=item[0] if item else None,
            item_y=item[1] if item else None,
        )

    # ---- lifecycle ------------------------------------------------------------------

    def scene_path(self, name: str) -> Path:
        return self.scenes_dir / f"{name}.state"

    def has_scene(self, name: str) -> bool:
        return self.scene_path(name).exists()

    def list_scenes(self) -> list[str]:
        return sorted(p.stem for p in self.scenes_dir.glob("*.state")) if self.scenes_dir.exists() else []

    def save_scene(self, name: str) -> Path:
        """Bookmark the current moment under `name` for exact, repeatable replays."""
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        path = self.scene_path(name)
        with open(path, "wb") as f:
            self.pb.save_state(f)
        return path

    def load_scene(self, name: str) -> RetroidState:
        with open(self.scene_path(name), "rb") as f:
            self.pb.load_state(f)
        self._ball_hist.clear()
        return self.state()

    def launch(self, hold: int = LAUNCH_HOLD) -> None:
        """Send the ball up off the paddle."""
        self.press("a", hold)

    def reset_to_play(self, name: str = "level1", force: bool = False) -> RetroidState:
        """Return to the level 1 start, navigating and bookmarking it once."""
        if self.has_scene(name) and not force:
            return self.load_scene(name)
        self._navigate_intro()
        self.save_scene(name)
        return self.state()

    def _navigate_intro(self) -> None:
        """Mash A through the opening, then let the paddle settle at the centre.

        The paddle slides in from the left and ignores input until it stops, so
        we detect the moment the sprites exist, stop pressing, and wait out the
        slide. Afterwards the ball must be resting on the paddle.
        """
        appeared = None
        for i in range(6000):
            if appeared is None:
                if i == INTRO_TICKS:
                    self.press("start", INTRO_START_HOLD)
                elif i > INTRO_TICKS and i % INTRO_A_PERIOD < INTRO_A_HOLD:
                    self.press("a", INTRO_A_HOLD)
            self.step(1)
            if appeared is None and self.paddle_x() is not None and self.ball() is not None:
                appeared = i
            if appeared is not None and i - appeared >= READY_DELAY:
                break
        if appeared is None:
            raise RuntimeError("opening did not reach the level; is this the right ROM?")
        if self.ball() is None:
            raise RuntimeError("the ball is not resting on the paddle after the opening")

    def close(self) -> None:
        self.pb.stop()

    def __enter__(self) -> "RetroidAdapter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
