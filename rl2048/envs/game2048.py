import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Game2048Env(gym.Env):
    """
    Gymnasium-compatible 2048 environment.

    - Actions: 0=up, 1=down, 2=left, 3=right
    - Observation: (size, size) int32 grid of tile values
    - Reward: sum of merged tile values produced by the move
    - Terminated: when max tile >= target
    - Truncated: when no further moves are possible
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, size: int = 4, target: int = 2048, spawn_prob_2: float = 0.9, render_mode: str | None = None):
        super().__init__()
        self.size = int(size)
        self.target = int(target)
        self.spawn_prob_2 = float(spawn_prob_2)
        self.render_mode = render_mode

        # 4 directions
        self.action_space = spaces.Discrete(4)
        # Conservative upper bound for tile values
        self.observation_space = spaces.Box(low=0, high=2 ** 16, shape=(self.size, self.size), dtype=np.int32)

        self.board: np.ndarray | None = None
        self.score: int = 0
        # Track previous transition details for observation augmentation
        self.prev_action: int = -1
        self.prev_moved: bool = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.prev_action = -1
        self.prev_moved = False
        self._add_tile()
        self._add_tile()
        info = {
            "score": self.score,
            "prev_action": self.prev_action,
            "prev_moved": self.prev_moved,
        }
        return self.board.copy(), info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(f"Invalid action: {action}")
        assert self.board is not None

        # valid actions mask before applying the action
        valid_before = self._valid_actions()

        moved, merge_score = self._move(action)
        reward = merge_score
        self.score += reward

        if moved:
            self._add_tile()

        max_tile = int(self.board.max())
        terminated = max_tile >= self.target
        truncated = not terminated and not self._can_move()

        # valid actions after the transition
        valid_after = self._valid_actions()

        # record previous action/effect for downstream wrappers/agents
        self.prev_action = int(action)
        self.prev_moved = bool(moved)

        info = {
            "score": self.score,
            "moved": bool(moved),
            "max_tile": max_tile,
            "valid_actions": valid_before.astype(bool),
            "valid_actions_next": valid_after.astype(bool),
            # expose previous action/effect as part of info for next-state processing
            "prev_action": self.prev_action,
            "prev_moved": self.prev_moved,
        }
        return self.board.copy(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "human" or self.render_mode is None:
            assert self.board is not None
            print("+" + "------+" * self.size)
            for r in range(self.size):
                row = "|".join(f"{int(v):^6}" if v > 0 else "      " for v in self.board[r])
                print("|" + row + "|")
                print("+" + "------+" * self.size)
            print(f"Score: {self.score}\n")

    # --- Internal helpers ---
    def _add_tile(self):
        assert self.board is not None
        empty_positions = np.argwhere(self.board == 0)
        if empty_positions.size == 0:
            return
        idx = self.np_random.integers(0, len(empty_positions))
        y, x = empty_positions[idx]
        value = 2 if self.np_random.random() < self.spawn_prob_2 else 4
        self.board[y, x] = value

    def _move(self, action: int) -> tuple[bool, int]:
        """Apply a move. Returns (moved, merge_score)."""
        assert self.board is not None

        # Map action to rotation count such that we always merge left
        # 0=up -> rot 1, 1=down -> rot 3, 2=left -> rot 0, 3=right -> rot 2
        rot_map = {0: 1, 1: 3, 2: 0, 3: 2}
        k = rot_map[int(action)]
        rotated = np.rot90(self.board, k)

        moved_any = False
        merge_score = 0
        new_rotated = rotated.copy()

        for i in range(self.size):
            row = rotated[i, :]
            compressed = row[row != 0]
            merged = []
            j = 0
            L = len(compressed)
            while j < L:
                if j + 1 < L and compressed[j] == compressed[j + 1]:
                    val = int(compressed[j]) * 2
                    merged.append(val)
                    merge_score += val
                    j += 2
                else:
                    merged.append(int(compressed[j]))
                    j += 1
            merged = np.array(merged, dtype=np.int32)
            pad = np.zeros(self.size - len(merged), dtype=np.int32)
            new_row = np.concatenate([merged, pad], axis=0)
            new_rotated[i, :] = new_row
            if not np.array_equal(new_row, row):
                moved_any = True

        # rotate back
        self.board = np.rot90(new_rotated, (4 - k) % 4)
        return moved_any, merge_score

    def _can_move(self) -> bool:
        assert self.board is not None
        if (self.board == 0).any():
            return True
        # any adjacent equal tiles?
        for axis in [0, 1]:
            arr = self.board if axis == 1 else self.board.T
            if np.any(arr[:, :-1] == arr[:, 1:]):
                return True
        return False

    def _valid_actions(self) -> np.ndarray:
        """Return boolean mask of valid actions for current board."""
        assert self.board is not None
        mask = np.zeros(4, dtype=bool)
        # Try each direction; valid if board would change
        for a in range(4):
            if self._would_move(a):
                mask[a] = True
        return mask

    def _would_move(self, action: int) -> bool:
        assert self.board is not None
        # simulate merge-left after rotation and see if any row changes
        rot_map = {0: 1, 1: 3, 2: 0, 3: 2}
        k = rot_map[int(action)]
        rotated = np.rot90(self.board, k)

        def merge_left(row: np.ndarray) -> np.ndarray:
            compressed = row[row != 0]
            merged = []
            j = 0
            L = len(compressed)
            while j < L:
                if j + 1 < L and compressed[j] == compressed[j + 1]:
                    merged.append(int(compressed[j]) * 2)
                    j += 2
                else:
                    merged.append(int(compressed[j]))
                    j += 1
            merged = np.array(merged, dtype=np.int32)
            pad = np.zeros(self.size - len(merged), dtype=np.int32)
            return np.concatenate([merged, pad], axis=0)

        for i in range(self.size):
            new_row = merge_left(rotated[i, :])
            if not np.array_equal(new_row, rotated[i, :]):
                return True
        return False

