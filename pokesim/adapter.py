"""Run Pokémon Red under PyBoy and expose it as a simple game object.

This is the only module that knows about PyBoy, ROM files and the game's RAM
layout. Everything downstream (encoders, rewards, the fly brain) talks to
`PokemonAdapter` in plain Python.

Addresses are for Pokémon Red (USA/Europe). They match the pret/pokered
disassembly and the ROM's SHA1, both recorded in README.md. We read them
directly instead of asking PyBoy to load `pokered.sym`, which prints a warning
per unsupported label.

A note on the opening: the game spends its first several thousand frames in
cinematics that advance only while A is held. `reset_to_overworld` walks through
them once and caches a save state, so later resets are instant and repeatable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyboy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROM = PROJECT_ROOT / "roms" / "pokered.gb"
DEFAULT_SYMBOLS = PROJECT_ROOT / "reference" / "pokered" / "pokered.sym"
SCENES_DIR = PROJECT_ROOT / "roms" / "scenes"

BUTTONS = ("a", "b", "start", "select", "up", "down", "left", "right")

# Map id $26. The game opens in the player's bedroom (2F), a safe, walkable place.
REDS_HOUSE_2F = 0x26

# Party monster layout in Gen 1: 44-byte records, HP at offset 1, max HP at $22.
PARTY_STRIDE = 44
PARTY_HP_OFFSET = 1
PARTY_MAXHP_OFFSET = 0x22

ADDR = {
    "map": 0xD35E,
    "y": 0xD361,
    "x": 0xD362,
    "facing": 0xC109,
    "moving_direction": 0xD528,
    "in_battle": 0xD057,
    "text_box": 0xD125,
    "menu_item": 0xCC26,
    "party_count": 0xD163,
    "party_mons": 0xD16B,
    "battle_hp": 0xD015,
    "battle_max_hp": 0xD023,
    "enemy_hp": 0xCFE6,
    "enemy_max_hp": 0xCFF4,
    "badges": 0xD356,
    "player_name": 0xD158,
    "num_sprites": 0xD4E1,
    "sprites1": 0xC100,   # 16 sprites x 16 bytes: picture id, screen y/x, facing...
    "sprites2": 0xC200,   # 16 sprites x 16 bytes: map y/x, movement state...
}

INTRO_TICKS = 300          # logo and title card
INTRO_START_HOLD = 6       # a tap can miss the title menu; hold briefly
INTRO_MENU_TICKS = 120
INTRO_A_HOLD = 6
INTRO_ROUNDS = 500         # dialogue rounds that walk the opening to the bedroom
INTRO_ROUND_TICKS = 14
SETTLE_CYCLES = 8          # A/B rounds that clear whatever box the opening left open


def load_symbols(path: str | Path = DEFAULT_SYMBOLS) -> dict[str, int]:
    """Every name in a pokered.sym file mapped to its address.

    The file is one `BB:AAAA name` per label. Parsing it ourselves keeps PyBoy
    from warning about the sublabels it refuses to load.
    """
    names: dict[str, int] = {}
    path = Path(path)
    if not path.exists():
        return names
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2 or ":" not in parts[0]:
            continue
        _, _, addr = parts[0].partition(":")
        try:
            names[parts[1]] = int(addr, 16)
        except ValueError:
            continue
    return names


@dataclass(frozen=True)
class GameState:
    """One moment of the game, as the fly's encoder and reward see it."""

    map_id: int
    x: int
    y: int
    facing: int
    in_battle: bool
    text_box: int
    menu_item: int
    party_count: int
    party_hp: tuple[int, ...]
    party_max_hp: tuple[int, ...]
    player_hp: int
    player_max_hp: int
    enemy_hp: int
    enemy_max_hp: int
    badges: int

    @property
    def in_overworld(self) -> bool:
        return not self.in_battle

    @property
    def party_hp_fraction(self) -> float:
        total, full = sum(self.party_hp), sum(self.party_max_hp)
        return total / full if full else 1.0

    @property
    def enemy_hp_fraction(self) -> float:
        return self.enemy_hp / self.enemy_max_hp if self.enemy_max_hp else 1.0


class PokemonAdapter:
    """A headless Pokémon Red you can step, press buttons on and read state from.

    `reset_to_overworld` returns the game to the moment the player first stands
    in their bedroom; `save_scene`/`load_scene` bookmark any other moment (a town
    with NPCs, a battle) so experiments restart from the exact same spot. Scenes
    live in `roms/scenes/` next to the ROM and are not committed.
    """

    def __init__(self, rom: str | Path = DEFAULT_ROM, symbols: str | Path | None = DEFAULT_SYMBOLS,
                 scenes_dir: str | Path | None = None, window: str = "null", scale: int = 3):
        self.rom = Path(rom)
        if not self.rom.exists():
            raise FileNotFoundError(f"no ROM at {self.rom}; see README.md")
        self.labels = load_symbols(symbols) if symbols else {}
        self.scenes_dir = Path(scenes_dir) if scenes_dir else SCENES_DIR
        self.pb = pyboy.PyBoy(str(self.rom), window=window, scale=scale)

    # ---- memory ---------------------------------------------------------------------

    def read(self, name_or_addr: str | int) -> int:
        if isinstance(name_or_addr, int):
            addr = name_or_addr
        else:
            addr = ADDR.get(name_or_addr, self.labels.get(name_or_addr, -1))
        if addr < 0:
            raise KeyError(f"unknown symbol: {name_or_addr}")
        return int(self.pb.memory[addr])

    def read16(self, addr: int) -> int:
        return (self.read(addr) << 8) | self.read(addr + 1)

    # ---- stepping and input ---------------------------------------------------------

    def step(self, frames: int = 1) -> None:
        self.pb.tick(frames)

    def press(self, button: str, frames: int = 6) -> None:
        """Hold `button` for `frames` frames, then release. Taps of one frame are
        often swallowed by the game's input debounce, so a short hold is the unit."""
        name = button.lower()
        if name not in BUTTONS:
            raise ValueError(f"button must be one of {BUTTONS}, not {button!r}")
        self.pb.button_press(name)
        self.step(frames)
        self.pb.button_release(name)

    # ---- observation ----------------------------------------------------------------

    def screen(self, gray: bool = False) -> np.ndarray:
        """The 144x160 screen as RGB uint8, or luminance uint8 if `gray`."""
        rgba = np.asarray(self.pb.screen.ndarray)
        rgb = rgba[..., :3]
        if not gray:
            return rgb.copy()
        return np.asarray(rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32), np.uint8)

    def state(self) -> GameState:
        party_count = self.read("party_count")
        hp, max_hp = [], []
        for i in range(party_count):
            base = ADDR["party_mons"] + i * PARTY_STRIDE
            hp.append(self.read16(base + PARTY_HP_OFFSET))
            max_hp.append(self.read16(base + PARTY_MAXHP_OFFSET))
        in_battle = self.read("in_battle") != 0
        return GameState(
            map_id=self.read("map"),
            x=self.read("x"),
            y=self.read("y"),
            facing=self.read("facing"),
            in_battle=in_battle,
            text_box=self.read("text_box"),
            menu_item=self.read("menu_item"),
            party_count=party_count,
            party_hp=tuple(hp),
            party_max_hp=tuple(max_hp),
            player_hp=self.read16(ADDR["battle_hp"]) if in_battle else 0,
            player_max_hp=self.read16(ADDR["battle_max_hp"]) if in_battle else 0,
            enemy_hp=self.read16(ADDR["enemy_hp"]) if in_battle else 0,
            enemy_max_hp=self.read16(ADDR["enemy_max_hp"]) if in_battle else 0,
            badges=bin(self.read("badges")).count("1"),
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

    def load_scene(self, name: str) -> GameState:
        with open(self.scene_path(name), "rb") as f:
            self.pb.load_state(f)
        return self.state()

    def reset_to_overworld(self, force_navigate: bool = False) -> GameState:
        """Return to the bedroom, navigating and bookmarking the opening once."""
        if self.has_scene("overworld") and not force_navigate:
            return self.load_scene("overworld")
        self._navigate_intro()
        self.save_scene("overworld")
        return self.state()

    def _navigate_intro(self) -> None:
        """Walk the opening to the bedroom, then make the player walkable.

        The opening is a fixed script, so we mash A for a fixed number of rounds
        rather than stopping when `wCurMap` first reads REDS_HOUSE_2F: that value
        passes through the map id during the cinematics, long before the bedroom
        is real. Afterwards a text box is often still open (mashing A can also
        trigger the SNES in the room), so `_settle` closes it.
        """
        self.step(INTRO_TICKS)
        self.press("start", INTRO_START_HOLD)
        self.step(INTRO_MENU_TICKS)
        self.press("a", INTRO_A_HOLD)
        for _ in range(INTRO_ROUNDS):
            self.press("a", INTRO_A_HOLD)
            self.step(INTRO_ROUND_TICKS)
        self._settle()
        if self.read("map") != REDS_HOUSE_2F:
            raise RuntimeError("opening did not reach the bedroom; is this the right ROM?")

    def _settle(self) -> None:
        """Close any leftover text box and confirm the player can actually walk.

        A and B never move the player, so probing with a short step and undoing
        it leaves the starting tile unchanged.
        """
        for _ in range(SETTLE_CYCLES):
            self.press("a", INTRO_A_HOLD)
            self.press("b", INTRO_A_HOLD)
            self.step(20)
            before = (self.read("x"), self.read("y"))
            self.press("down", 16)
            self.step(4)
            if (self.read("x"), self.read("y")) != before:
                self.press("up", 16)
                self.step(4)
                return
        raise RuntimeError("player could not move after the opening")

    def close(self) -> None:
        self.pb.stop()

    def __enter__(self) -> "PokemonAdapter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
