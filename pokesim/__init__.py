"""Tools for letting the fly body play Pokémon Red.

`PokemonAdapter` is the only module that knows about PyBoy, ROM files and the
game's RAM layout. Encoders, rewards and the fly brain talk to it in plain
Python: screen frames in, buttons out, named state for free.
"""
from .adapter import ADDR, BUTTONS, GameState, PokemonAdapter, load_symbols
from .encoder_b import OracleEncoder, OracleTarget
from .encoding import Dataset, record, sample_screen, walk_until

__all__ = ["PokemonAdapter", "GameState", "OracleEncoder", "OracleTarget",
           "Dataset", "record", "sample_screen", "walk_until",
           "ADDR", "BUTTONS", "load_symbols"]
