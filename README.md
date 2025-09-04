**RL2048**

Train agents to play 2048 using a clean Gymnasium environment with Stable-Baselines3 or a lightweight discrete SAC. This repo is structured as a small Python package to make it easy to extend and contribute new ideas.

Key features:
- Gymnasium-compatible `Game2048Env` with valid-action masks and informative `info`.
- SB3 training with observation/action wrappers and optional shaping rewards.
- Minimal discrete SAC trainer (PyTorch) for a pure-Py approach.
- Hydra configs under `conf/` for reproducible runs.

Quick start
- Python 3.9+
- Install dependencies: `pip install -e .[sb3,torch,dev]` (or use `requirements.txt`).
- Run SB3 training: `rl2048-train-sb3`.
- Play with a saved model: `rl2048-use-model --path sb3_sac_model_corner_network.zip`.
- Human play (arrow keys): `rl2048-play`.

Project layout
- `rl2048/`: package with `envs/` and `wrappers/`.
- `conf/`: Hydra configs (`env.yaml`, `train.yaml`, `train_sb3.yaml`).
- `scripts/`: standalone training script for discrete SAC.
- `outputs/`, `wandb/`: suggested output/log dirs (gitignored).

Install
- Editable install: `pip install -e .`.
- With extras: `pip install -e .[sb3,torch,dev]`.
- Or: `pip install -r requirements.txt`.

Commands
- `rl2048-train-sb3`: trains SAC from SB3 using `conf/train_sb3.yaml`.
- `rl2048-train-discrete`: trains the discrete SAC in `scripts/train_sac.py` using `conf/train.yaml`.
- `rl2048-play`: plays human with arrow keys using `conf/env.yaml`.
- `rl2048-use-model`: runs a saved `.zip` (SB3) or `.pt` (discrete) model.

Config
Hydra reads configs from `conf/`:
- `env.yaml`: board size/target/spawn params for play.
- `train.yaml`: discrete SAC hyperparameters.
- `train_sb3.yaml`: SB3 hyperparameters and reward shaping.

Environment API
- Observation: `np.int32` board `(size, size)` with tile values.
- Actions: `0=up, 1=down, 2=left, 3=right`.
- Reward: sum of merge values for the move (wrappers may reshape).
- Terminated: when max tile >= `target`.
- Truncated: when no further moves are possible.

Development
- Lint: `ruff check .`.
- Test: `pytest`.
- Package metadata: `pyproject.toml`.

Contributing
See `CONTRIBUTING.md` for guidelines and a short checklist for adding new features (wrappers, agents, configs). PRs welcome!

