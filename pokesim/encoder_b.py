"""The oracle encoder: drive the fly's visual channels from exact game state.

Where the visual encoder (encoder A) sees only pixels, this reads the RAM truth
-- the player's tile, every map object's on-screen position, HP, battle flags --
and turns it into the same `FeatureDetectors` channels. It is the ceiling the
visual encoder gets measured against, and the teacher we train it from.

The mapping mirrors sshfighter: an "opponent" at (dx, size) in screen pixels and
a 0..1 threat level. Positions are egocentric and in pixels -- sprite screen X/Y
from `wSpriteStateData1`, minus the player's -- because that is exactly what the
retina would see. In the overworld the opponent is the nearest map object; in a
battle it is the enemy Pokémon, which sits on the right with a size that tracks
how much of it is left.
"""
from __future__ import annotations

from dataclasses import dataclass

from flybrain.eyes import FeatureDetectors

from .adapter import ADDR

SPRITE_STRIDE = 16
SPRITE1_PICTURE_ID = 0
SPRITE1_IMAGE_INDEX = 2    # $ff when the sprite is off screen
SPRITE1_Y = 4              # screen Y in pixels (4 above the tile centre)
SPRITE1_X = 6              # screen X in pixels
OFFSCREEN = 0xFF
OBJECT_SIZE = 16.0         # a map-object sprite is 16x16 pixels
THREAT_RANGE = 96.0        # pixels at which an object stops feeling close
BATTLE_DX = 64.0           # the enemy sits on the right of the battle screen


@dataclass(frozen=True)
class OracleTarget:
    """What the fly would react to: an object `dx, dy` screen pixels away (dx
    negative is left, dy negative is up), `size` pixels across, with a 0..1
    threat level. `side` is the label a visual encoder can be scored against."""

    dx: float
    dy: float
    size: float
    threat: float
    kind: str

    @property
    def side(self) -> str:
        return "L" if self.dx < 0 else "R"

    @property
    def distance(self) -> float:
        return (self.dx * self.dx + self.dy * self.dy) ** 0.5


def objects(adapter) -> list[tuple[int, int, float]]:
    """(dx, dy, distance) in pixels for every active map object, player out.

    Sprites are struct 0 (the player) then 1..`wNumSprites` (see ram/wram.asm);
    a zero picture id or an off-screen image index means the slot is unused.
    """
    pbase = ADDR["sprites1"]
    px, py = adapter.read(pbase + SPRITE1_X), adapter.read(pbase + SPRITE1_Y)
    found = []
    for i in range(1, adapter.read("num_sprites") + 1):
        b = ADDR["sprites1"] + i * SPRITE_STRIDE
        if adapter.read(b + SPRITE1_PICTURE_ID) == 0 or adapter.read(b + SPRITE1_IMAGE_INDEX) == OFFSCREEN:
            continue
        dx = adapter.read(b + SPRITE1_X) - px
        dy = adapter.read(b + SPRITE1_Y) - py
        found.append((dx, dy, (dx * dx + dy * dy) ** 0.5))
    return found


def target(adapter) -> OracleTarget | None:
    """The nearest map object (or the battle enemy) in the fly's frame, or None."""
    state = adapter.state()
    if state.in_battle:
        return OracleTarget(dx=BATTLE_DX, dy=0.0, size=OBJECT_SIZE * (0.5 + state.enemy_hp_fraction),
                            threat=1.0 - state.enemy_hp_fraction, kind="enemy")
    nearest = None
    for dx, dy, dist in objects(adapter):
        if nearest is None or dist < nearest[2]:
            nearest = (dx, dy, dist)
    if nearest is None:
        return None
    dx, dy, dist = nearest
    return OracleTarget(dx=dx, dy=dy, size=OBJECT_SIZE, threat=max(0.0, 1.0 - dist / THREAT_RANGE),
                        kind="object")


class OracleEncoder:
    """Turns exact state into `FeatureDetectors.inject(...)` calls."""

    def __init__(self, brain, adapter):
        self.adapter = adapter
        self.detectors = FeatureDetectors(brain)

    @property
    def channels(self) -> dict:
        return self.detectors.cells

    def objects(self):
        return objects(self.adapter)

    def target(self):
        return target(self.adapter)

    def inject(self) -> list:
        """The `inject` list for one `brain.step`, or [] when nothing is in view."""
        t = self.target()
        if t is None:
            return []
        return self.detectors.inject(opp=(t.dx, t.size), threat=t.threat)
