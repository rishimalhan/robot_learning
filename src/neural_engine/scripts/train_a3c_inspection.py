#!/usr/bin/env python3
"""A3C-style training script for the InspectionEnv."""

# External

import argparse
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
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
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_log_std_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        mean = torch.clip(torch.tanh(self.policy_mean(feat)), -1.0, 1.0)
        log_std = torch.clamp(self.policy_log_std_head(feat), -5.0, 2.0)
        std = torch.exp(log_std)
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
    cfg: A3CConfig,
    episodes: int = 3,
    max_steps: int = 1000,
) -> float:
    device = next(model.parameters()).device
    model.eval()
    total_return = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        episode_return = 0.0
        for _ in range(max_steps):
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                mean, _, _ = model.forward(state_t)
            action = mean.squeeze(0).cpu().numpy()
            state, reward, done, terminated, _ = env.step(action)
            episode_return += reward
            if done or terminated:
                break
        total_return += episode_return
    return total_return / episodes


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
        for step in range(1, cfg.total_steps + 1):
            states_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
            dist = model.dist(states_t)
            actions_t = dist.rsample()
            log_probs = dist.log_prob(actions_t).sum(-1)
            entropy = dist.entropy().sum(-1)
            values = model.value(states_t)

            next_states, rewards, dones, _ = batcher.step(
                actions_t.detach().cpu().numpy()
            )
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE)
            next_states_t = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
            next_values = model.value(next_states_t).detach()
            targets = rewards_t + cfg.gamma * next_values * (1.0 - dones_t)
            advantages = targets - values

            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            entropy_loss = -entropy.mean()
            loss = (
                actor_loss
                + cfg.value_coef * critic_loss
                + cfg.entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            states = next_states
            if step % 100 == 0:
                eval_return = evaluate_policy(eval_env, model, cfg)
                print(
                    f"Step {step}/{cfg.total_steps} | Loss {loss.item():.4f} | Eval return {eval_return:.3f}"
                )
            if step % 1000 == 0:
                torch.save(model.state_dict(), CHECKPOINT_PATH)
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
    finally:
        eval_env.close()
        batcher.close()


def visualize_policy(model: ActorCritic, cfg: A3CConfig):
    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    env = make_env(
        publish_pointcloud=True,
        visualize=True,
        point_stride=cfg.point_stride,
    )
    state, info = env.reset()
    done = False
    print("Starting visualization run...")
    try:
        while not done:
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                mean, _, _ = model.forward(state_t)
            action = mean.squeeze(0).cpu().numpy()
            state, reward, done, terminated, info = env.step(action)
            done = done or terminated
    except KeyboardInterrupt:
        print("Visualization interrupted by user.")
    finally:
        env.close()
    print("Visualization finished.")


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
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
        sample_env.close()
        visualize_policy(model, cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    rospy.init_node("train_a3c_inspection", disable_signals=True)
    try:
        main()
    except KeyboardInterrupt:
        print("Shutdown requested by user.")
