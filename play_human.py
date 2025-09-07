import curses
from typing import Optional

import hydra
from omegaconf import DictConfig

from rl2048.envs.game2048 import Game2048Env


def draw_board(stdscr, board, score: int):
    stdscr.clear()
    rows, cols = board.shape

    # Terminal size
    h, w = stdscr.getmaxyx()

    # Choose cell width based on largest value for better fit
    try:
        max_val = int(board.max())
    except Exception:
        max_val = 0
    cell_w = max(4, len(str(max(2, max_val))) + 2)

    # Compute required dimensions
    board_width = 1 + cols * (cell_w + 1)  # e.g. +------+-...+
    total_height = rows * 2 + 3  # rows lines + borders + score/instructions

    # If too small, prompt user to resize
    if board_width > w or total_height > h:
        msg1 = "Window too small for board"
        msg2 = f"Need at least {board_width}x{total_height}, have {w}x{h}"
        if h > 0:
            stdscr.addstr(0, 0, msg1[: max(0, w)])
        if h > 1:
            stdscr.addstr(1, 0, msg2[: max(0, w)])
        stdscr.refresh()
        return

    # Center the board
    top = max(0, (h - total_height) // 2)
    left = max(0, (w - board_width) // 2)

    # Draw border and cells (clip to width if needed, though we checked bounds)
    horiz = "+" + ("-" * cell_w + "+") * cols
    for r in range(rows):
        # horizontal border
        stdscr.addstr(top + r * 2, left, horiz)
        # values row
        line = "|".join(
            f"{int(v):^{cell_w}}" if v > 0 else " " * cell_w for v in board[r]
        )
        stdscr.addstr(top + r * 2 + 1, left, "|" + line + "|")

    stdscr.addstr(top + rows * 2, left, horiz)
    stdscr.addstr(top + rows * 2 + 1, left, f"Score: {score}")
    stdscr.addstr(top + rows * 2 + 2, left, "Arrows to move, 'q' to quit")
    stdscr.refresh()


def play_loop(stdscr, env: Game2048Env, seed: Optional[int] = None):
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    obs, info = env.reset(seed=seed)
    score = int(info.get("score", 0))
    draw_board(stdscr, obs, score)

    key_to_action = {
        curses.KEY_UP: 0,
        curses.KEY_DOWN: 1,
        curses.KEY_LEFT: 2,
        curses.KEY_RIGHT: 3,
    }

    while True:
        ch = stdscr.getch()
        # Redraw on resize to adapt layout
        if ch == curses.KEY_RESIZE:
            draw_board(stdscr, obs, score)
            continue
        if ch in (ord("q"), ord("Q")):
            break
        if ch not in key_to_action:
            continue

        action = key_to_action[ch]
        obs, reward, terminated, truncated, info = env.step(action)
        score = int(info.get("score", 0))
        draw_board(stdscr, obs, score)

        if terminated or truncated:
            msg = "You win!" if terminated else "No more moves. Game over."
            stdscr.addstr(0, 2, msg + " Press any key...")
            stdscr.refresh()
            stdscr.getch()
            obs, info = env.reset(seed=seed)
            score = int(info.get("score", 0))
            draw_board(stdscr, obs, score)


@hydra.main(config_path="./conf", config_name="env", version_base=None)
def main(cfg: DictConfig):
    env_cfg = cfg.env
    env = Game2048Env(
        size=int(env_cfg.size),
        target=int(env_cfg.target),
        spawn_prob_2=float(env_cfg.spawn_prob_2),
        render_mode=str(env_cfg.render_mode),
    )
    seed = cfg.get("seed")
    curses.wrapper(play_loop, env=env, seed=seed)


if __name__ == "__main__":
    main()
