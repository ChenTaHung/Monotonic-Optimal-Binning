"""Core algorithms and utilities: PAVA, constraints, merging, utils."""

from .constraints import BinningConstraints
from .pava import PAVA
from .merge import Block, merge_adjacent, as_blocks
from . import utils

__all__ = [
    "BinningConstraints",
    "PAVA",
    "Block",
    "merge_adjacent",
    "as_blocks",
    "utils",
]
