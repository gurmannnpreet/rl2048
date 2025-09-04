Contributing to RL2048

Thanks for considering a contribution! This repo aims to be small and approachable so new ideas can land quickly. Here’s how to get set up and make changes that are easy to review.

Setup
- Python 3.9+ recommended.
- Install: `pip install -e .[sb3,torch,dev]` (or `pip install -r requirements.txt`).
- Run tests: `pytest`.
- Lint (optional): `ruff check .`.

Repo structure
- `rl2048/`: Python package.
  - `envs/game2048.py`: the Gymnasium `Game2048Env`.
  - `wrappers/sb3.py`: observation/action wrappers and `make_env` for SB3.
- `scripts/`: runnable training scripts (e.g., discrete SAC).
- `conf/`: Hydra configs (`env.yaml`, `train.yaml`, `train_sb3.yaml`).
- `tests/`: minimal sanity tests.

Common tasks
- Add a wrapper: put it in `rl2048/wrappers/` and export from `__init__.py`.
- Add a config option: thread it through the relevant script and document it.
- Add a training script: place under `scripts/` or provide a small CLI in the package and add an entrypoint in `pyproject.toml`.

Style
- Keep changes focused and minimal.
- Prefer small, composable functions and modules.
- Avoid introducing heavy dependencies; use extras in `pyproject.toml` if needed.

Pull requests
- Describe the motivation and scope clearly.
- Include usage examples or mention config changes.
- If touching training logic, run a short sanity check (a few thousand steps) to verify nothing crashes.

Thanks!

