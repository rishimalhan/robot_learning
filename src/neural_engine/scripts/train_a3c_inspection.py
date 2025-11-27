#!/usr/bin/env python3
"""A3C-style training script for the InspectionEnv."""

# External

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import defaultdict
import rospy

# Internal

from neural_engine.rl_environment import InspectionEnv


CHECKPOINT_PATH = Path(__file__).resolve().parent / "inspection_a3c_latest.pt"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MAX_GRAD_NORM = 1.0


def make_env(
    publish_pointcloud: bool, visualize: bool, point_stride: int
) -> InspectionEnv:
    return InspectionEnv(
        publish_pointcloud=publish_pointcloud,
        visualize=visualize,
        point_stride=point_stride,
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_log_std_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        mean = self.policy_mean(feat)
        var = F.softplus(self.policy_log_std_head(feat)) + 1e-6
        std = torch.sqrt(var)
        value = self.value_head(feat).squeeze(-1)
        return mean, std, value

    def dist(self, states: torch.Tensor) -> Normal:
        mean, std, _ = self.forward(states)
        return Normal(mean, std)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        _, _, value = self.forward(states)
        return value


@dataclass
class A3CConfig:
    num_envs: int = 8
    total_steps: int = 20000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    visualize_policy: bool = False
    publish_pointcloud: bool = False
    point_stride: int = 4
    n_step_horizon: int = 5


class EnvBatch:
    def __init__(self, cfg: A3CConfig, action_dim: int):
        self.envs = [
            make_env(
                publish_pointcloud=cfg.publish_pointcloud,
                visualize=False,
                point_stride=cfg.point_stride,
            )
            for _ in range(cfg.num_envs)
        ]
        self._action_dim = action_dim

    def reset(self) -> np.ndarray:
        return np.stack([env.reset()[0] for env in self.envs], axis=0)

    def step(self, actions: np.ndarray):
        next_states, rewards, dones = [], [], []
        infos = []
        for env, act in zip(self.envs, actions):
            obs, reward, done, terminated, info = env.step(act)
            done_flag = done or terminated
            if done_flag:
                obs, info = env.reset()
            next_states.append(obs)
            rewards.append(reward)
            dones.append(done_flag)
            infos.append(info)
        return (
            np.stack(next_states, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()


def evaluate_policy(
    env: InspectionEnv,
    model: ActorCritic,
    episodes: int = 3,
    max_steps: int = 10,
) -> str:
    device = next(model.parameters()).device
    model.eval()
    breakdown_history = {
        "coverage": [],
        "penalty": [],
        "orientation_violation": [],
        "oob_dist": [],
    }
    for episode_idx in range(episodes):
        state, _ = env.reset()
        episode_breakdown = defaultdict(float)
        for _ in range(max_steps):
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                mean, _, _ = model.forward(state_t)
            action = mean.squeeze(0).cpu().numpy()
            state, reward, done, terminated, info = env.step(action)
            rb = info.get("reward_breakdown")
            episode_breakdown["coverage"] += float(rb.get("coverage"))
            episode_breakdown["penalty"] += float(rb.get("penalty"))
            episode_breakdown["steps"] += 1
            episode_breakdown["orientation_violation"] += float(
                rb.get("orientation_violation")
            )
            episode_breakdown["oob_dist"] += float(rb.get("oob_dist"))
            if done or terminated:
                break
        for key in ("coverage", "penalty", "orientation_violation", "oob_dist"):
            breakdown_history[key].append(episode_breakdown[key])
        message = (
            "[Eval] Episode "
            f"{episode_idx + 1}/{episodes} | steps={episode_breakdown['steps']} "
            f"| coverage={float(episode_breakdown['coverage']):.3f} "
            f"| penalty={float(episode_breakdown['penalty']):.3f} "
            f"| orientation_violation={float(episode_breakdown['orientation_violation']):.3f} "
            f"| oob_dist={float(episode_breakdown['oob_dist']):.3f}"
        )
        return message
    return "N/A"


def train(cfg: A3CConfig):
    sample_env = make_env(
        publish_pointcloud=cfg.publish_pointcloud,
        visualize=False,
        point_stride=cfg.point_stride,
    )
    obs_dim = sample_env.observation_space.shape[0]
    action_dim = sample_env.action_space.shape[0]
    sample_env.close()

    eval_env = make_env(
        publish_pointcloud=False,
        visualize=False,
        point_stride=cfg.point_stride,
    )

    model = ActorCritic(obs_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    batcher = EnvBatch(cfg, action_dim)
    states = batcher.reset()

    try:
        step = 0
        while step < cfg.total_steps:
            rollout_log_probs: List[torch.Tensor] = []
            rollout_entropies: List[torch.Tensor] = []
            rollout_values: List[torch.Tensor] = []
            rollout_rewards: List[torch.Tensor] = []
            rollout_dones: List[torch.Tensor] = []
            rollout_len = 0

            while rollout_len < cfg.n_step_horizon and step < cfg.total_steps:
                states_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
                dist = model.dist(states_t)
                actions_t = dist.rsample()
                log_probs = dist.log_prob(actions_t).sum(-1)
                entropy = dist.entropy().sum(-1)
                values = model.value(states_t)

                next_states, rewards, dones, _ = batcher.step(
                    actions_t.detach().cpu().numpy()
                )

                rollout_log_probs.append(log_probs)
                rollout_entropies.append(entropy)
                rollout_values.append(values)
                rollout_rewards.append(
                    torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
                )
                rollout_dones.append(
                    torch.tensor(dones, dtype=torch.float32, device=DEVICE)
                )

                states = next_states
                rollout_len += 1
                step += 1

            next_states_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
            next_values = model.value(next_states_t).detach()

            returns = next_values
            actor_loss = torch.zeros(1, device=DEVICE)
            critic_loss = torch.zeros(1, device=DEVICE)
            entropy_loss = torch.zeros(1, device=DEVICE)

            for t in reversed(range(rollout_len)):
                mask = 1.0 - rollout_dones[t]
                returns = rollout_rewards[t] + cfg.gamma * returns * mask
                advantages = returns - rollout_values[t]

                actor_loss += -(rollout_log_probs[t] * advantages.detach()).mean()
                critic_loss += advantages.pow(2).mean()
                entropy_loss += -rollout_entropies[t].mean()

            rollout_scale = float(max(rollout_len, 1))
            loss = (
                actor_loss / rollout_scale
                + cfg.value_coef * (critic_loss / rollout_scale)
                + cfg.entropy_coef * (entropy_loss / rollout_scale)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                eval_return = evaluate_policy(eval_env, model)
                print(
                    f"Step {step}/{cfg.total_steps} | Eval: {eval_return}\n"
                )
            if step % 1000 == 0:
                torch.save(model.state_dict(), CHECKPOINT_PATH)
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
    finally:
        eval_env.close()
        batcher.close()


def main():
    parser = argparse.ArgumentParser(description="Train A3C agent on InspectionEnv")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--publish-cloud", action="store_true")
    args = parser.parse_args()

    cfg = A3CConfig(
        total_steps=args.steps,
        num_envs=args.envs,
        visualize_policy=args.visualize,
        publish_pointcloud=args.publish_cloud,
        point_stride=4,
    )

    if args.visualize:
        sample_env = make_env(
            publish_pointcloud=cfg.publish_pointcloud,
            visualize=False,
            point_stride=cfg.point_stride,
        )
        model = ActorCritic(
            sample_env.observation_space.shape[0], sample_env.action_space.shape[0]
        ).to(DEVICE)
        sample_env.close()
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
        viz_env = make_env(
            publish_pointcloud=True,
            visualize=True,
            point_stride=cfg.point_stride,
        )
        try:
            evaluate_policy(viz_env, model, episodes=1, max_steps=20)
        finally:
            viz_env.close()
    else:
        train(cfg)


if __name__ == "__main__":
    rospy.init_node("train_a3c_inspection", disable_signals=True)
    try:
        main()
    except KeyboardInterrupt:
        print("Shutdown requested by user.")
