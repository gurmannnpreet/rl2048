import argparse
import os
import time
from typing import Optional

import numpy as np
import torch

from rl2048.envs.game2048 import Game2048Env


def load_sb3_model(path: str):
    try:
        from stable_baselines3 import SAC
    except Exception as e:
        raise RuntimeError("stable-baselines3 is required to load SB3 models. Install with: pip install stable-baselines3") from e
    return SAC.load(path, device="cpu")


def make_sb3_env():
    # Reuse wrappers from the package to ensure action/obs compatibility
    from rl2048.wrappers.sb3 import Log2ObsWrapper, ContinuousActionWrapper, InvalidMovePenaltyWrapper

    env = Game2048Env(render_mode="human")
    # Keep same shaping as training; visual impact is only on rewards, not rendering
    env = InvalidMovePenaltyWrapper(env)
    env = Log2ObsWrapper(env)
    # Prefer keeping max tile at bottom-right in action mapping
    env = ContinuousActionWrapper(env, preferred=[1, 3])
    return env


def make_discrete_env():
    return Game2048Env(render_mode="human")


def evaluate_sb3(model, steps: int = 1000, seed: Optional[int] = None, delay: float = 0.1):
    env = make_sb3_env()
    obs, info = env.reset(seed=seed)
    env.render()
    total_reward = 0.0
    for t in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        env.render()
        time.sleep(delay)
        if terminated or truncated:
            break
    print(f"SB3 play: steps={t+1}, shaped_return={total_reward:.2f}, score={info.get('score')}, max_tile={info.get('max_tile')}")


def evaluate_discrete_sac(path: str, steps: int = 1000, seed: Optional[int] = None, delay: float = 0.1):
    # Import agent utilities from our training script
    from train_sac import DiscreteSAC, preprocess_obs
    
    def prefer_mask(mask: np.ndarray, corner: str = "bottom_right") -> np.ndarray:
        if mask is None:
            return mask
        mapping = {
            "top_left": [0, 2],
            "top_right": [0, 3],
            "bottom_left": [1, 2],
            "bottom_right": [1, 3],
        }
        pref = mapping.get(corner, mapping["bottom_right"])
        mask = mask.astype(bool).copy()
        if mask.any():
            pref_mask = np.zeros_like(mask, dtype=bool)
            for a in pref:
                if 0 <= a < mask.size:
                    pref_mask[a] = mask[a]
            if pref_mask.any():
                return pref_mask
        return mask

    payload = torch.load(path, map_location="cpu")
    cfg = payload.get("config", {})
    hidden_dim = int(cfg.get("agent", {}).get("hidden_dim", 128))

    env = make_discrete_env()
    num_actions = int(env.action_space.n)
    obs, info = env.reset(seed=seed)
    obs_vec = preprocess_obs(obs, info.get("prev_action"), info.get("prev_moved"))
    obs_dim = int(obs_vec.shape[0])

    agent = DiscreteSAC(
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        device="cpu",
    )
    # Load only the policy for inference
    agent.policy.load_state_dict(payload["policy"])
    agent.policy.eval()

    env.render()
    total_reward = 0.0
    for t in range(steps):
        mask = info.get("valid_actions") if isinstance(info.get("valid_actions"), np.ndarray) else None
        mask = prefer_mask(mask, corner="bottom_right")
        a = agent.act(obs_vec, greedy=True, mask=mask)
        next_obs, reward, terminated, truncated, info = env.step(a)
        total_reward += float(reward)
        env.render()
        time.sleep(delay)
        if terminated or truncated:
            break
        obs_vec = preprocess_obs(next_obs, info.get("prev_action"), info.get("prev_moved"))

    print(f"Discrete SAC play: steps={t+1}, return={total_reward:.2f}, score={info.get('score')}, max_tile={info.get('max_tile')}")


def main():
    parser = argparse.ArgumentParser(description="Play 2048 using a saved model and print gameplay.")
    parser.add_argument("--path", type=str, default=None, help="Path to model file (.zip for SB3, .pt for discrete SAC). Defaults to sb3_sac_model.zip if present else model.pt")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps to run")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay seconds between renders")
    args = parser.parse_args()

    path = args.path
    if path is None:
        if os.path.exists("sb3_sac_model_corner_network.zip"):
            path = "sb3_sac_model_corner_network.zip"
        elif os.path.exists("model.pt"):
            path = "model.pt"
        else:
            raise FileNotFoundError("No model path provided and neither sb3_sac_model.zip nor model.pt found in CWD.")

    if path.endswith(".zip"):
        model = load_sb3_model(path)
        evaluate_sb3(model, steps=args.steps, seed=args.seed, delay=args.delay)
    elif path.endswith(".pt"):
        evaluate_discrete_sac(path, steps=args.steps, seed=args.seed, delay=args.delay)
    else:
        raise ValueError("Unknown model file type. Use .zip for SB3 or .pt for discrete SAC.")


if __name__ == "__main__":
    main()
