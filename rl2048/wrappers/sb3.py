from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl2048.envs.game2048 import Game2048Env


class Log2ObsWrapper(gym.ObservationWrapper):
    """Convert board to flattened log2-scaled float32 vector in [0,1],
    and append previous action one-hot (4) + moved flag (1).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        size = getattr(env, "size", 4)
        # board features + 4 one-hot for prev action + 1 moved flag
        self.obs_len = size * size + 5
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_len,), dtype=np.float32)

    def observation(self, obs):
        # board -> log2 scale
        x = obs.astype(np.float32)
        out = np.zeros_like(x, dtype=np.float32)
        nz = x > 0
        out[nz] = np.log2(x[nz]) / 16.0
        board_vec = out.reshape(-1)

        # previous action one-hot and moved flag from base env state
        base = self.env.unwrapped
        prev_action = getattr(base, "prev_action", -1)
        prev_moved = 1.0 if bool(getattr(base, "prev_moved", False)) else 0.0
        one_hot = np.zeros(4, dtype=np.float32)
        if isinstance(prev_action, (int, np.integer)) and 0 <= int(prev_action) < 4:
            one_hot[int(prev_action)] = 1.0

        return np.concatenate([board_vec, one_hot, np.array([prev_moved], dtype=np.float32)], axis=0)


class ContinuousActionWrapper(gym.ActionWrapper):
    """Map a continuous 1D action in [-1,1] to a VALID discrete move.

    - Reads the base `Game2048Env` valid-actions mask before applying the step.
    - Projects the continuous value into the set of currently-valid actions.
    - Prevents repeated invalid moves in eval/training.
    """

    def __init__(self, env: gym.Env, preferred: list[int] | None = None):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Preferred directions to keep max tile in target corner (e.g., bottom_right -> [1,3])
        self.preferred = set(int(a) for a in (preferred or []))

    def action(self, action):
        a = float(action[0]) if isinstance(action, (list, tuple, np.ndarray)) else float(action)
        # Access the base env to compute valid actions for current board
        base = self.env.unwrapped
        valid = None
        if hasattr(base, "_valid_actions"):
            try:
                valid = base._valid_actions().astype(bool)
            except Exception:
                valid = None
        # Fallback to naive binning if mask unavailable
        if valid is None or valid.sum() == 0:
            bins = [-0.5, 0.0, 0.5]
            return int(np.digitize(a, bins=bins))

        valid_idxs = np.flatnonzero(valid)
        # If any preferred directions are currently valid, restrict choices to them
        if len(self.preferred) > 0:
            pref = [i for i in valid_idxs if i in self.preferred]
            if len(pref) > 0:
                valid_idxs = np.array(pref, dtype=int)
        # Map a in [-1,1] to rank in [0, len(valid_idxs)-1]
        p = (a + 1.0) / 2.0
        k = int(np.clip(np.floor(p * len(valid_idxs)), 0, len(valid_idxs) - 1))
        return int(valid_idxs[k])


class InvalidMovePenaltyWrapper(gym.Wrapper):
    """Adds a small penalty when a move does not change the board (info['moved']=False)."""

    def __init__(self, env: gym.Env, penalty: float = -0.5, scale: float = 0.1):
        super().__init__(env)
        self.penalty = float(penalty)
        self.scale = float(scale)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = reward * self.scale
        if not bool(info.get("moved", True)):
            shaped += self.penalty
        return obs, shaped, terminated, truncated, info


class CornerBonusWrapper(gym.Wrapper):
    """Adds a bonus when the maximum tile is located at a target corner.

    - Target corner: one of {top_left, top_right, bottom_left, bottom_right}
    - Bonus magnitude: `bonus_coef * (log2(max_tile)/16)`; 0 if no tiles.
    """

    def __init__(self, env: gym.Env, corner: str = "bottom_right", bonus_coef: float = 0.1):
        super().__init__(env)
        self.corner = str(corner)
        self.bonus_coef = float(bonus_coef)

    def _corner_idx(self):
        size = getattr(self.env.unwrapped, "size", 4)
        mapping = {
            "top_left": (0, 0),
            "top_right": (0, size - 1),
            "bottom_left": (size - 1, 0),
            "bottom_right": (size - 1, size - 1),
        }
        return mapping.get(self.corner, mapping["bottom_right"])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            board = self.env.unwrapped.board
            if board is not None:
                max_val = int(board.max())
                if max_val > 0:
                    y, x = self._corner_idx()
                    if int(board[y, x]) == max_val:
                        bonus = self.bonus_coef * (np.log2(max_val) / 16.0)
                        reward = float(reward) + float(bonus)
        except Exception:
            pass
        return obs, reward, terminated, truncated, info


def make_env(cfg_env, reward_cfg) -> gym.Env:
    env = Game2048Env(
        size=int(cfg_env.size),
        target=int(cfg_env.target),
        spawn_prob_2=float(cfg_env.spawn_prob_2),
        render_mode=None,
    )
    env = InvalidMovePenaltyWrapper(env, penalty=float(reward_cfg.invalid_move_penalty), scale=float(reward_cfg.scale))
    # corner-based bonus shaping
    corner = str(getattr(reward_cfg, "corner", "bottom_right"))
    corner_bonus = float(getattr(reward_cfg, "corner_bonus", 0.0))
    if corner_bonus != 0.0:
        env = CornerBonusWrapper(env, corner=corner, bonus_coef=corner_bonus)
    env = Log2ObsWrapper(env)
    # Preferred directions by corner to keep max tile anchored
    corner_map = {
        "top_left": [0, 2],
        "top_right": [0, 3],
        "bottom_left": [1, 2],
        "bottom_right": [1, 3],
    }
    preferred = corner_map.get(corner, corner_map["bottom_right"])
    env = ContinuousActionWrapper(env, preferred=preferred)
    return env

