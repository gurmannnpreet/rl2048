import os
from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import gymnasium as gym
from gymnasium import spaces

from game_2048_env import Game2048Env


class Log2ObsWrapper(gym.ObservationWrapper):
    """Convert board to flattened log2-scaled float32 vector in [0,1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        size = getattr(env, "size", 4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(size * size,), dtype=np.float32)

    def observation(self, obs):
        x = obs.astype(np.float32)
        out = np.zeros_like(x, dtype=np.float32)
        nz = x > 0
        out[nz] = np.log2(x[nz]) / 16.0
        return out.reshape(-1)


class ContinuousActionWrapper(gym.ActionWrapper):
    """Map a continuous 1D action in [-1,1] to a VALID discrete move.

    - Reads the base `Game2048Env` valid-actions mask before applying the step.
    - Projects the continuous value into the set of currently-valid actions.
    - Prevents repeated invalid moves in eval/training.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

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


def make_env(cfg_env, reward_cfg) -> gym.Env:
    env = Game2048Env(
        size=int(cfg_env.size),
        target=int(cfg_env.target),
        spawn_prob_2=float(cfg_env.spawn_prob_2),
        render_mode=None,
    )
    env = InvalidMovePenaltyWrapper(env, penalty=float(reward_cfg.invalid_move_penalty), scale=float(reward_cfg.scale))
    env = Log2ObsWrapper(env)
    env = ContinuousActionWrapper(env)
    return env


def evaluate(model, env: gym.Env, seed=None, max_steps: int = 5000) -> Tuple[float, int]:
    obs, info = env.reset(seed=seed)
    ep_return = 0.0
    ep_len = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        ep_len += 1
        if terminated or truncated or ep_len >= max_steps:
            break
    return ep_return, ep_len


@hydra.main(config_path="/Users/gurmannnpreet/Documents/CODING/rl2048/conf", config_name="train_sb3", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Lazy import to give a clearer error if missing
    try:
        from stable_baselines3 import SAC
    except Exception as e:
        raise RuntimeError("stable-baselines3 is required: pip install stable-baselines3") from e

    env = make_env(cfg.env, cfg.reward)
    eval_env = make_env(cfg.env, cfg.reward)

    policy_kwargs = dict(net_arch=[int(cfg.agent.hidden_dim), int(cfg.agent.hidden_dim)])
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=float(cfg.agent.lr),
        buffer_size=int(cfg.buffer.capacity),
        batch_size=int(cfg.train.batch_size),
        tau=float(cfg.agent.tau),
        gamma=float(cfg.agent.gamma),
        train_freq=int(cfg.train.train_freq),
        gradient_steps=int(cfg.train.gradient_steps),
        ent_coef="auto" if cfg.agent.alpha is None else float(cfg.agent.alpha),
        verbose=1,
        tensorboard_log=os.getcwd(),
        seed=None if cfg.seed is None else int(cfg.seed),
        policy_kwargs=policy_kwargs,
        device="cpu",
    )

    total_timesteps = int(cfg.train.steps)
    model.learn(total_timesteps=total_timesteps, log_interval=10)

    # Save model in Hydra run dir
    save_path = os.path.join(os.getcwd(), "sb3_sac_model")
    model.save(save_path)
    print(f"Saved SB3 SAC model to: {save_path}.zip")

    # Quick greedy eval
    ret, length = evaluate(model, eval_env, seed=cfg.seed, max_steps=int(cfg.train.eval_max_steps))
    print(f"Greedy eval: return={ret:.2f}, len={length}")


if __name__ == "__main__":
    main()
