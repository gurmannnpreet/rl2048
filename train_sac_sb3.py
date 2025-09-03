import os
from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import gymnasium as gym
from gymnasium import spaces

from game_2048_env import Game2048Env
from stable_baselines3 import SAC



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
    size_dummy = getattr(env.unwrapped, "size", 4)
    corner_map = {
        "top_left": [0, 2],
        "top_right": [0, 3],
        "bottom_left": [1, 2],
        "bottom_right": [1, 3],
    }
    preferred = corner_map.get(corner, corner_map["bottom_right"])
    env = ContinuousActionWrapper(env, preferred=preferred)
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

    env = make_env(cfg.env, cfg.reward)
    eval_env = make_env(cfg.env, cfg.reward)

    # Optional Weights & Biases init (robust config handling)
    cfg_wb = getattr(cfg, "wandb", None)
    try:
        use_wandb = bool(cfg_wb and bool(getattr(cfg_wb, "enabled")))
    except Exception:
        use_wandb = False
    if use_wandb:
        try:
            import os as _os
            import wandb
            mode = getattr(cfg.wandb, "mode", None)
            if mode:
                _os.environ["WANDB_MODE"] = str(mode)
            wandb.init(
                project=str(getattr(cfg.wandb, "project", "rl2048")),
                entity=None if getattr(cfg.wandb, "entity", None) in (None, "null") else str(cfg.wandb.entity),
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True,
                reinit=True,
            )
            wandb_run = True
        except Exception as e:
            print(f"wandb init failed or not installed: {e}. Proceeding without wandb.")
            wandb_run = False
    else:
        wandb_run = False

    # Allow custom net_arch via config; fallback to [hidden_dim, hidden_dim]
    net_arch = None
    try:
        arch = cfg.agent.get("net_arch") if hasattr(cfg, "agent") else None
        if arch is not None:
            net_arch = [int(x) for x in arch]
    except Exception:
        net_arch = None
    if net_arch is None:
        net_arch = [int(cfg.agent.hidden_dim), int(cfg.agent.hidden_dim)]
    policy_kwargs = dict(net_arch=net_arch)
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

    # Periodic evaluation + wandb logging via callback
    from stable_baselines3.common.callbacks import BaseCallback

    class EvalAndWandbCallback(BaseCallback):
        def __init__(self, eval_env, eval_interval: int, eval_max_steps: int, seed=None, use_wandb=False):
            super().__init__()
            self.eval_env = eval_env
            self.eval_interval = int(eval_interval)
            self.eval_max_steps = int(eval_max_steps)
            self.seed = seed
            self.use_wandb = use_wandb
            # Optional psutil for system/process stats
            self._proc = None
            try:
                import psutil  # type: ignore
                self._psutil = psutil
                self._proc = psutil.Process()
                # prime CPU percent calculation
                self._proc.cpu_percent(interval=None)
            except Exception:
                self._psutil = None

        def _collect_stats(self):
            stats = {}
            try:
                if self._psutil is not None and self._proc is not None:
                    # System-level
                    stats["sys/cpu_percent"] = float(self._psutil.cpu_percent(interval=None))
                    try:
                        vm = self._psutil.virtual_memory()
                        stats["sys/mem_percent"] = float(vm.percent)
                    except Exception:
                        pass
                    try:
                        import os as _os
                        la = _os.getloadavg()
                        stats["sys/loadavg_1m"] = float(la[0])
                    except Exception:
                        pass
                    # Process-level
                    stats["proc/cpu_percent"] = float(self._proc.cpu_percent(interval=None))
                    try:
                        mem = self._proc.memory_info()
                        stats["proc/mem_rss_mb"] = float(mem.rss) / (1024.0 * 1024.0)
                    except Exception:
                        pass
            except Exception:
                pass
            return stats

        def _on_step(self) -> bool:
            if self.eval_interval > 0 and (self.num_timesteps % self.eval_interval == 0):
                ret, length = evaluate(self.model, self.eval_env, seed=self.seed, max_steps=self.eval_max_steps)
                alpha_val = None
                try:
                    if hasattr(self.model, "log_ent_coef") and self.model.log_ent_coef is not None:
                        alpha_val = float(np.exp(self.model.log_ent_coef.detach().cpu().numpy()))
                    elif hasattr(self.model, "ent_coef") and isinstance(self.model.ent_coef, (float, int)):
                        alpha_val = float(self.model.ent_coef)
                except Exception:
                    alpha_val = None
                msg = f"Step {self.num_timesteps} | eval_return={ret:.2f} | eval_len={length}"
                if alpha_val is not None:
                    msg += f" | alpha={alpha_val:.4f}"
                # Gather system/process stats
                stats = self._collect_stats()
                if stats:
                    msg += " | " + ", ".join(f"{k}={v:.2f}" for k, v in stats.items() if isinstance(v, (int, float)))
                print(msg, flush=True)
                if self.use_wandb:
                    try:
                        import wandb
                        data = {"eval/return": ret, "eval/len": length, "time/steps": int(self.num_timesteps)}
                        if alpha_val is not None:
                            data["sac/alpha"] = alpha_val
                        # Attach system/process stats under their keys
                        for k, v in stats.items():
                            data[k] = v
                        wandb.log(data, step=int(self.num_timesteps))
                    except Exception:
                        pass
            return True

    total_timesteps = int(cfg.train.steps)
    eval_interval = int(getattr(cfg.train, "eval_interval", 0))
    eval_max_steps = int(getattr(cfg.train, "eval_max_steps", 5000))
    callback = EvalAndWandbCallback(eval_env=eval_env, eval_interval=eval_interval, eval_max_steps=eval_max_steps, seed=cfg.seed, use_wandb=wandb_run)
    model.learn(total_timesteps=total_timesteps, log_interval=10, callback=callback)

    # Save model in Hydra run dir
    save_path = os.path.join(os.getcwd(), "sb3_sac_model_corner")
    model.save(save_path)
    print(f"Saved SB3 SAC model to: {save_path}.zip")
    if wandb_run:
        try:
            import wandb
            wandb.save(f"{save_path}.zip")
        except Exception:
            pass

    # Quick greedy eval
    ret, length = evaluate(model, eval_env, seed=cfg.seed, max_steps=int(cfg.train.eval_max_steps))
    print(f"Greedy eval: return={ret:.2f}, len={length}")
    if wandb_run:
        try:
            import wandb
            wandb.log({
                "eval/return": ret,
                "eval/len": length,
                "final_eval/return": ret,
                "final_eval/len": length,
                "time/steps": int(total_timesteps),
            }, step=int(total_timesteps))
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
