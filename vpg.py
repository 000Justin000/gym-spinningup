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


def model_update(target_model, source_model, tau=0.05):
    for target_param, source_param in zip(
        target_model.parameters(), source_model.parameters()
    ):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )


def greedy_inference(policy_model):
    env = gym.make("MsPacman-v4", render_mode="human")
    state_manager = StateManager(STACK_SIZE)
    obs, info = env.reset()
    state_manager.push(obs)

    list_reward = []
    while True:
        state = state_manager.get()
        with torch.no_grad():
            action = policy_model({"x": state})["x"].argmax(dim=-1).item()
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
    policy_model = setup_module(model_config["policy_model"]).to("mps")
    value_model = setup_module(model_config["value_model"]).to("mps")

    def select_action(state, policy_model):
        with torch.no_grad():
            distribution = Categorical(logits=policy_model({"x": state})["x"])
        action = distribution.sample()
        return action

    def reward_to_go(rewards, gamma):
        rtg = torch.zeros_like(rewards)
        rtg[-1] = rewards[-1]
        for i in reversed(range(len(rewards) - 1)):
            rtg[i] = rewards[i] + gamma * rtg[i + 1]
        return rtg

    env = gym.make("MsPacman-v4", render_mode=None)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=LR)
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    for episode in range(NUM_EPISODES):
        state_manager = StateManager(STACK_SIZE)

        obs, info = env.reset()
        state_manager.push(obs)

        # initialize reward to zero for the first step
        reward = 0
        state = state_manager.get()
        action = select_action(state, policy_model)
        trajectory = [(reward, state, action)]
        for step in range(MAX_NUM_STEPS):
            obs, reward, terminated, truncated, info = env.step(action)
            state_manager.push(obs)

            state = state_manager.get()
            action = select_action(state, policy_model)

            trajectory.append((reward, state, action))

            if terminated:
                break

        # vectorization
        list_reward, list_state, list_action = zip(*trajectory)
        rewards = torch.tensor(list_reward, dtype=torch.float32).to(DEVICE)  # [T]
        states = torch.cat(list_state, dim=0).to(DEVICE)  # [T, C, H, W]
        actions = torch.tensor(list_action, dtype=torch.long).to(DEVICE)  # [T]

        # compute log probability of actions
        distribution = Categorical(logits=policy_model({"x": states})["x"])
        log_probs = distribution.log_prob(actions)

        # compute loss
        loss = -torch.mean(log_probs * reward_to_go(rewards, GAMMA))

        # update policy model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=10.0)
        optimizer.step()

        total_reward = sum(list_reward)
        episode_length = len(trajectory)
        mean_loss = loss.item()

        writer.add_scalar("Episode/Reward", total_reward, episode)
        writer.add_scalar("Episode/Length", episode_length, episode)
        writer.add_scalar("Episode/AvgLoss", mean_loss, episode)


if __name__ == "__main__":
    main()
