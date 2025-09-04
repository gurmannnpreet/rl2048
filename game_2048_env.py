"""Compatibility shim.

Prefer importing `Game2048Env` from `rl2048.envs.game2048`.
This module re-exports it for backward compatibility.
"""

from rl2048.envs.game2048 import Game2048Env  # noqa: F401
