import click
import json
from utils.nn.module import setup_module

import torch
from torchvision import transforms
from torch.nn import functional as F

import gym
from time import sleep

import random

import copy

from collections import deque

from torch.utils.tensorboard import SummaryWriter
import os
import time

import numpy as np
import math
from torch.distributions import Categorical


NUM_EPISODES = 1000
MAX_NUM_STEPS = 1000_000  # infinity
NUM_OPTIM_STEPS = 1000
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100_000
STACK_SIZE = 4
LR = 0.0001
DEVICE = "mps"

log_dir = os.path.join("runs", "vpg_" + time.strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir)


class StateManager:
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),  # [H, W, C] -> [C, H, W]
                transforms.Grayscale(num_output_channels=1),  # [C, H, W] -> [1, H, W]
                transforms.Resize((84, 84)),  # [1, H, W] -> [1, 84, 84]
                transforms.Lambda(
                    lambda x: x.unsqueeze(0)
                ),  # [1, 84, 84] -> [1, 1, 84, 84]
                transforms.Lambda(
                    lambda x: x.to(DEVICE)
                ),  # [1, 1, 84, 84] -> [1, 1, 84, 84]
            ]
        )

    def push(self, observation):
        frame = self.preprocess(observation)
        while len(self.frames) < self.stack_size:
            self.frames.append(frame)
        self.frames.append(frame)

    def get(self):
        return torch.cat(list(self.frames), dim=1)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            torch.cat(states, dim=0),
            torch.tensor(actions).reshape(-1, 1).to(DEVICE),
            torch.tensor(rewards).reshape(-1, 1).to(DEVICE),
            torch.cat(next_states, dim=0),
            torch.tensor(dones).reshape(-1, 1).to(DEVICE),
        )


def greedy_inference(actor_critic):
    env = gym.make("MsPacman-v4", render_mode="human")
    state_manager = StateManager(STACK_SIZE)
    obs, info = env.reset()
    state_manager.push(obs)

    list_reward = []
    while True:
        state = state_manager.get()
        with torch.no_grad():
            action = actor_critic({"x": state})["x_policy"].argmax(dim=-1).item()
        obs, reward, terminated, truncated, info = env.step(action)
        state_manager.push(obs)
        list_reward.append(reward)
        if terminated:
            break
    env.close()
    return sum(list_reward)


@click.command()
@click.option("--model_config", type=click.Path(exists=True), required=True)
def main(model_config: str):
    with open(model_config, "r") as f:
        model_config = json.load(f)
    actor_critic = setup_module(model_config).to("mps")

    def select_action(state, actor_critic):
        with torch.no_grad():
            batch = actor_critic({"x": state})
            distribution = Categorical(logits=batch["x_policy"])
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob

    def future_discounted_sum(rewards, gamma):
        rtg = torch.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            rtg[i] = rewards[i] + gamma * (rtg[i + 1] if i + 1 < len(rewards) else 0)
        return rtg

    env = gym.make("MsPacman-v4", render_mode=None)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=LR)
    for episode in range(NUM_EPISODES):
        state_manager = StateManager(STACK_SIZE)

        obs, info = env.reset()
        state_manager.push(obs)

        trajectory = []
        for step in range(MAX_NUM_STEPS):
            state = state_manager.get()
            action = select_action(state, actor_critic)
            obs, reward, terminated, truncated, info = env.step(action)
            state_manager.push(obs)

            trajectory.append((state, action, reward))

            if terminated:
                break

        # vectorization
        list_state, list_action, list_reward = zip(*trajectory)
        states = torch.cat(list_state, dim=0).to(DEVICE)  # [T, C, H, W]
        actions = torch.tensor(list_action, dtype=torch.long).to(DEVICE)  # [T]
        rewards = torch.tensor(list_reward, dtype=torch.float32).to(DEVICE)  # [T]

        # compute log probability of actions
        batch = actor_critic({"x": states})
        distribution = Categorical(logits=batch["x_policy"])
        log_probs = distribution.log_prob(actions)

        # compute loss
        rtg = future_discounted_sum(rewards, GAMMA)
        val = batch["x_value"]
        adv = rtg - val
        normalized_adv = (adv - adv.mean()) / adv.std()
        policy_loss = -torch.mean(log_probs * normalized_adv)
        value_loss = F.mse_loss(rtg, val)

        # update policy model
        optimizer.zero_grad()
        loss = policy_loss + value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=10.0)
        optimizer.step()

        total_reward = sum(list_reward)
        episode_length = len(trajectory)
        mean_loss = loss.item()

        writer.add_scalar("Episode/Reward", total_reward, episode)
        writer.add_scalar("Episode/Length", episode_length, episode)
        writer.add_scalar("Episode/PolicyLoss", policy_loss.item(), episode)
        writer.add_scalar("Episode/ValueLoss", value_loss.item(), episode)


if __name__ == "__main__":
    main()
