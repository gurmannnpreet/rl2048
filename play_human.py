import curses
from typing import Optional

import hydra
from omegaconf import DictConfig

from rl2048.envs.game2048 import Game2048Env


def draw_board(stdscr, board, score: int):
    stdscr.clear()
    rows, cols = board.shape
    top = 1
    left = 2
    cell_w = 6
    # Draw border and cells
    for r in range(rows):
        # horizontal border
        stdscr.addstr(top + r * 2, left, "+" + ("-" * cell_w + "+") * cols)
        # values row
        line = "|".join(f"{int(v):^{cell_w}}" if v > 0 else " " * cell_w for v in board[r])
        stdscr.addstr(top + r * 2 + 1, left, "|" + line + "|")
    stdscr.addstr(top + rows * 2, left, "+" + ("-" * cell_w + "+") * cols)
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
