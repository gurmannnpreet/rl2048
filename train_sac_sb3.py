import os
from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import SAC
from rl2048.wrappers.sb3 import make_env


def evaluate(model, env, seed=None, max_steps: int = 5000) -> Tuple[float, int]:
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


@hydra.main(config_path="conf", config_name="train_sb3", version_base=None)
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
    # Load model from path if specified, else create new
    load_path = getattr(cfg.agent, "load_model_path", None)
    if load_path:
        model = SAC.load(load_path, env=env, device="cpu")
        print(f"Loaded SAC model from: {load_path}")
    else:
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
    save_path = os.path.join(os.getcwd(), "sb3_sac_model_corner_network")
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
