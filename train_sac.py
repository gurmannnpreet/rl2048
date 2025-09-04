import math
import os
import random
from dataclasses import dataclass
from typing import Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from rl2048.envs.game2048 import Game2048Env


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_obs(obs: np.ndarray, prev_action: int | None = None, prev_moved: bool | None = None) -> np.ndarray:
    # Map board values to log2 scale in [0, 1]
    # 0 stays 0; others -> log2(value)/16 for 2^16 upper
    x = obs.astype(np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    nz = x > 0
    out[nz] = np.log2(x[nz]) / 16.0
    board_vec = out.reshape(-1)

    # Append prev action (one-hot of 4) and moved flag (1)
    one_hot = np.zeros(4, dtype=np.float32)
    if prev_action is not None and isinstance(prev_action, (int, np.integer)) and 0 <= int(prev_action) < 4:
        one_hot[int(prev_action)] = 1.0
    moved_flag = np.array([1.0 if bool(prev_moved) else 0.0], dtype=np.float32)

    return np.concatenate([board_vec, one_hot, moved_flag], axis=0)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ReplayBuffer:
    capacity: int
    obs_dim: int
    device: torch.device
    num_actions: int

    def __post_init__(self):
        self.ptr = 0
        self.size = 0
        self.obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.mask_buf = np.zeros((self.capacity, self.num_actions), dtype=np.bool_)
        self.next_mask_buf = np.zeros((self.capacity, self.num_actions), dtype=np.bool_)

    def add(self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: bool, mask: np.ndarray, next_mask: np.ndarray):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)
        self.mask_buf[self.ptr] = mask
        self.next_mask_buf[self.ptr] = next_mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs_buf[idxs], device=self.device)
        act = torch.as_tensor(self.act_buf[idxs], device=self.device)
        rew = torch.as_tensor(self.rew_buf[idxs], device=self.device)
        next_obs = torch.as_tensor(self.next_obs_buf[idxs], device=self.device)
        done = torch.as_tensor(self.done_buf[idxs], device=self.device)
        mask = torch.as_tensor(self.mask_buf[idxs], device=self.device)
        next_mask = torch.as_tensor(self.next_mask_buf[idxs], device=self.device)
        return obs, act, rew, next_obs, done, mask, next_mask


class DiscreteSAC:
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        alpha: float | None = None,
        target_entropy: float | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau

        # Critics output Q-values for all actions
        self.q1 = MLP(obs_dim, num_actions, hidden_dim).to(self.device)
        self.q2 = MLP(obs_dim, num_actions, hidden_dim).to(self.device)
        self.q1_target = MLP(obs_dim, num_actions, hidden_dim).to(self.device)
        self.q2_target = MLP(obs_dim, num_actions, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy outputs logits over actions
        self.policy = MLP(obs_dim, num_actions, hidden_dim).to(self.device)

        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Entropy temperature
        if alpha is None:
            # automatic tuning
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self._alpha = None
            if target_entropy is None:
                # Aim near maximum entropy
                self.target_entropy = 0.9 * math.log(num_actions)
            else:
                self.target_entropy = float(target_entropy)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self._alpha = float(alpha)
            self.target_entropy = None

    @property
    def alpha(self) -> torch.Tensor:
        if self._alpha is not None:
            return torch.tensor(self._alpha, device=self.device)
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, greedy: bool = False, mask: np.ndarray | None = None) -> int:
        self.policy.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.policy(x)
            if mask is not None:
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).view(1, -1)
                logits = logits.masked_fill(~mask_t, -1e9)
            if greedy:
                a = torch.argmax(logits, dim=-1)
                return int(a.item())
            probs = F.softmax(logits, dim=-1)
            a = torch.distributions.Categorical(probs=probs).sample()
            return int(a.item())

    def update(self, batch, updates: int):
        obs, act, rew, next_obs, done, mask, next_mask = batch

        # Compute target Q
        with torch.no_grad():
            # Next state policy and entropy
            next_logits = self.policy(next_obs)
            next_logits = next_logits.masked_fill(~next_mask.bool(), -1e9)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_probs = next_log_probs.exp()

            q1_next = self.q1_target(next_obs)
            q2_next = self.q2_target(next_obs)
            q_next = torch.min(q1_next, q2_next)

            # V(s') = sum_a pi(a|s')[Q(s',a) - alpha * log pi(a|s')]
            v_next = (next_probs * (q_next - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rew + (1.0 - done) * self.gamma * v_next

        # Critic loss
        q1 = self.q1(obs).gather(1, act.view(-1, 1)).squeeze(1)
        q2 = self.q2(obs).gather(1, act.view(-1, 1)).squeeze(1)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 5.0)
        self.q_optimizer.step()

        # Policy loss: E_s [ sum_a pi(a|s) * (alpha*log pi(a|s) - minQ(s,a)) ]
        logits = self.policy(obs)
        logits = logits.masked_fill(~mask.bool(), -1e9)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        with torch.no_grad():
            q_min = torch.min(self.q1(obs), self.q2(obs))
        policy_loss = (probs * (self.alpha.detach() * log_probs - q_min)).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.policy_optimizer.step()

        # Temperature loss (if auto)
        alpha_loss_val = torch.tensor(0.0, device=self.device)
        if self.alpha_optimizer is not None:
            entropy = -(probs * log_probs).sum(dim=-1).mean().detach()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy))
            alpha_loss_val = alpha_loss.mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss_val.backward()
            self.alpha_optimizer.step()

        # Soft update targets
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_targ.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_targ.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss_val.item()),
        }


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = "cpu"  # keep CPU-only for small Macs
    set_seed(cfg.seed)

    env_cfg = cfg.env
    env = Game2048Env(
        size=int(env_cfg.size),
        target=int(env_cfg.target),
        spawn_prob_2=float(env_cfg.spawn_prob_2),
        render_mode=None,
    )
    # separate eval env to avoid interfering with training episode
    eval_env = Game2048Env(
        size=int(env_cfg.size),
        target=int(env_cfg.target),
        spawn_prob_2=float(env_cfg.spawn_prob_2),
        render_mode=None,
    )

    num_actions = int(env.action_space.n)
    obs, info = env.reset(seed=cfg.seed)
    obs_vec = preprocess_obs(obs, info.get("prev_action"), info.get("prev_moved"))
    obs_dim = int(obs_vec.shape[0])

    agent = DiscreteSAC(
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden_dim=int(cfg.agent.hidden_dim),
        gamma=float(cfg.agent.gamma),
        tau=float(cfg.agent.tau),
        lr=float(cfg.agent.lr),
        alpha=None if cfg.agent.alpha is None else float(cfg.agent.alpha),
        target_entropy=None if cfg.agent.target_entropy is None else float(cfg.agent.target_entropy),
        device=device,
    )

    buf = ReplayBuffer(capacity=int(cfg.buffer.capacity), obs_dim=obs_dim, device=torch.device(device), num_actions=num_actions)

    steps = int(cfg.train.steps)
    start_steps = int(cfg.train.start_steps)
    batch_size = int(cfg.train.batch_size)
    eval_interval = int(cfg.train.eval_interval)

    ep_return = 0.0
    ep_len = 0
    episode = 0
    updates = 0

    for t in range(1, steps + 1):
        # Select action
        if t < start_steps:
            action = np.random.randint(0, num_actions)
        else:
            def prefer_mask(mask: np.ndarray | None, corner: str = "bottom_right") -> np.ndarray | None:
                if mask is None:
                    return None
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

            corner_pref = str(cfg.reward.get("corner", "bottom_right"))
            mask_for_act = info.get("valid_actions") if isinstance(info.get("valid_actions"), np.ndarray) else None
            mask_for_act = prefer_mask(mask_for_act, corner=corner_pref)
            action = agent.act(obs_vec, greedy=False, mask=mask_for_act)

        next_obs, reward, terminated, truncated, info_next = env.step(action)
        done = bool(terminated or truncated)
        next_obs_vec = preprocess_obs(next_obs, info_next.get("prev_action"), info_next.get("prev_moved"))
        # Reward shaping
        reward_scaled = float(reward) * float(cfg.reward.scale)
        if not bool(info_next.get("moved", True)):
            reward_scaled += float(cfg.reward.invalid_move_penalty)
        # Corner bonus: encourage max tile to stay in a target corner
        try:
            corner_bonus = float(cfg.reward.get("corner_bonus", 0.0))
        except Exception:
            corner_bonus = 0.0
        if corner_bonus != 0.0:
            try:
                corner = str(cfg.reward.get("corner", "bottom_right"))
                size = int(env.size)
                mapping = {
                    "top_left": (0, 0),
                    "top_right": (0, size - 1),
                    "bottom_left": (size - 1, 0),
                    "bottom_right": (size - 1, size - 1),
                }
                y, x = mapping.get(corner, mapping["bottom_right"])
                max_val = int(next_obs.max())
                if max_val > 0 and int(next_obs[y, x]) == max_val:
                    reward_scaled += corner_bonus * (float(np.log2(max_val)) / 16.0)
            except Exception:
                pass
        mask_curr = info.get("valid_actions") if isinstance(info.get("valid_actions"), np.ndarray) else None
        mask_curr = prefer_mask(mask_curr, corner=corner_pref)
        if mask_curr is None:
            mask_curr = np.ones(num_actions, dtype=bool)
        mask_next = info_next.get("valid_actions") if isinstance(info_next.get("valid_actions"), np.ndarray) else info_next.get("valid_actions_next")
        mask_next = prefer_mask(mask_next, corner=corner_pref)
        if mask_next is None:
            mask_next = np.ones(num_actions, dtype=bool)

        buf.add(obs_vec, action, reward_scaled, next_obs_vec, done, mask_curr, mask_next)

        obs_vec = next_obs_vec
        info = info_next
        ep_return += float(reward)
        ep_len += 1

        if done:
            episode += 1
            obs, info = env.reset(seed=cfg.seed)
            obs_vec = preprocess_obs(obs, info.get("prev_action"), info.get("prev_moved"))
            print(f"Episode {episode} | return={ep_return:.1f} | len={ep_len}")
            ep_return = 0.0
            ep_len = 0

        # Update agent
        if buf.size >= batch_size:
            batch = buf.sample(batch_size)
            stats = agent.update(batch, updates)
            updates += 1

        # Periodic eval (greedy run)
        if t % eval_interval == 0:
            print("Evaluating (greedy)...", flush=True)
            # For eval, use current state's valid actions where available by reconstructing mask on the fly is complex;
            # rely on env to provide valid actions during eval loop via info.
            eval_return, eval_len = evaluate(eval_env, agent, seed=cfg.seed, max_steps=int(cfg.train.get('eval_max_steps', 5000)))
            print(f"Step {t} | eval_return={eval_return:.1f} | eval_len={eval_len} | alpha={agent.alpha.item():.4f}")

    # Save final model to Hydra run directory
    save_path = os.path.join(os.getcwd(), "model.pt")
    payload = {
        "policy": agent.policy.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "q1_target": agent.q1_target.state_dict(),
        "q2_target": agent.q2_target.state_dict(),
        "policy_optimizer": agent.policy_optimizer.state_dict(),
        "q_optimizer": agent.q_optimizer.state_dict(),
        "alpha": float(agent.alpha.item()),
        "log_alpha": None if getattr(agent, "log_alpha", None) is None else float(agent.log_alpha.detach().cpu().item()),
        "alpha_optimizer": None if getattr(agent, "alpha_optimizer", None) is None else agent.alpha_optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(payload, save_path)
    print(f"Saved final model to: {save_path}")


def evaluate(env: Game2048Env, agent: DiscreteSAC, seed=None, max_steps: int | None = None) -> Tuple[float, int]:
    obs, info = env.reset(seed=seed)
    obs_vec = preprocess_obs(obs, info.get("prev_action"), info.get("prev_moved"))
    ep_return = 0.0
    ep_len = 0
    while True:
        mask = info.get("valid_actions") if isinstance(info.get("valid_actions"), np.ndarray) else None
        mask = prefer_mask(mask, corner=str(cfg.reward.get("corner", "bottom_right")))
        a = agent.act(obs_vec, greedy=True, mask=mask)
        next_obs, reward, terminated, truncated, info = env.step(a)
        ep_return += float(reward)
        ep_len += 1
        if terminated or truncated or (max_steps is not None and ep_len >= max_steps):
            break
        obs_vec = preprocess_obs(next_obs, info.get("prev_action"), info.get("prev_moved"))
    return ep_return, ep_len


if __name__ == "__main__":
    main()
