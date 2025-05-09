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


SLEEP_TIME = 0.0
NUM_EPISODES = 100000
MAX_NUM_STEPS = 1000
EPSILON = 0.2
GAMMA = 0.999
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
STACK_SIZE = 1
LR = 0.0001
DEVICE = "mps"

log_dir = os.path.join("runs", "dqn_" + time.strftime("%Y%m%d-%H%M%S"))
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
        return torch.cat(list(self.frames), dim=-1)


class ReplayBuffer:
    def __init__(self, capacity, default_batch_size):
        self.buffer = deque(maxlen=capacity)
        self.default_batch_size = default_batch_size

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.default_batch_size
        transitions = random.choices(self.buffer, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            torch.cat(states, dim=0),
            torch.tensor(actions).reshape(-1, 1).to(DEVICE),
            torch.tensor(rewards).reshape(-1, 1).to(DEVICE),
            torch.cat(next_states, dim=0),
            torch.tensor(dones).reshape(-1, 1).to(DEVICE),
        )


def soft_update(target_model, source_model, tau=0.05):
    for target_param, source_param in zip(
        target_model.parameters(), source_model.parameters()
    ):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )


@click.command()
@click.option("--model_config", type=click.Path(exists=True), required=True)
def main(model_config: str):
    with open(model_config, "r") as f:
        model_config = json.load(f)
    policy_model = setup_module(model_config).to("mps")
    target_model = copy.deepcopy(policy_model).eval()

    # env = gym.make("MsPacman-v4", render_mode="human")
    env = gym.make("MsPacman-v4")

    def select_action(state, policy_model):
        if random.random() < EPSILON:
            return env.action_space.sample()
        else:
            return policy_model({"x": state})["x"].argmax(dim=-1).item()

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=LR)
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, BATCH_SIZE)
    for episode in range(NUM_EPISODES):
        state_manager = StateManager(STACK_SIZE)
        obs, info = env.reset()
        state_manager.push(obs)
        list_reward = []
        list_q_est = []
        list_loss = []
        for step in range(MAX_NUM_STEPS):
            state = state_manager.get()
            action = select_action(state, policy_model)
            obs, reward, terminated, truncated, info = env.step(action)
            state_manager.push(obs)
            next_state = state_manager.get()
            buffer.push(state, action, reward, next_state, terminated)

            # training
            (
                batch_state,
                batch_action,
                batch_reward,
                batch_next_state,
                batch_terminated,
            ) = buffer.sample()
            q = policy_model({"x": batch_state})["x"]
            with torch.no_grad():
                next_q = target_model({"x": batch_next_state})["x"]

            # compute loss
            q_est = torch.gather(q, dim=-1, index=batch_action)
            q_tgt = batch_reward + GAMMA * next_q.max(dim=-1, keepdim=True).values * ~batch_terminated
            loss = F.mse_loss(q_est, q_tgt)

            # update NN
            optimizer.zero_grad()
            loss.backward()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=10.0)
            optimizer.step()

            soft_update(target_model, policy_model, tau=0.05)

            list_reward.append(reward)
            list_q_est.append(q_est.mean().item())
            list_loss.append(loss.item())

            if terminated:
                break

            if SLEEP_TIME > 0:
                sleep(SLEEP_TIME)

        total_reward = sum([GAMMA**i * reward for i, reward in enumerate(list_reward)])
        mean_q = sum(list_q_est) / len(list_q_est)
        mean_loss = sum(list_loss) / len(list_loss)

        writer.add_scalar("Episode/Reward", total_reward, episode)
        writer.add_scalar("Episode/MeanQ", mean_q, episode)
        writer.add_scalar("Episode/AvgLoss", mean_loss, episode)

        print(
            f"Episode {episode} finished with total_reward {total_reward:.4f}, q_est {mean_q:.4f}, loss {mean_loss:.4f}"
        )


if __name__ == "__main__":
    main()
