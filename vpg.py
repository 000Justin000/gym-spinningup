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


NUM_EPISODES = 1000
MAX_NUM_STEPS = 1000_000 # infinity
NUM_OPTIM_STEPS = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 300
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

    def select_action(state, policy_model, episode):
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(
            -1.0 * episode / EPSILON_DECAY
        )
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_model({"x": state})["x"].argmax(dim=-1).item()
        return action

    def transform_reward(raw_reward, eaten):
        if eaten:
            return -math.log(20, 1000)
        elif raw_reward > 0:
            return math.log(raw_reward, 1000)
        else:
            return raw_reward

    env = gym.make("MsPacman-v4", render_mode=None)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=LR)
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    for episode in range(NUM_EPISODES):
        state_manager = StateManager(STACK_SIZE)

        obs, info = env.reset()
        state_manager.push(obs)

        list_reward = []
        for step in range(MAX_NUM_STEPS):
            state = state_manager.get()
            action = select_action(state, policy_model, episode)

            # prev_info = info
            obs, reward, terminated, truncated, info = env.step(action)
            # reward = transform_reward(raw_reward, info["lives"] < prev_info["lives"])

            state_manager.push(obs)
            next_state = state_manager.get()
            buffer.push(state, action, reward, next_state, terminated)

            list_reward.append(reward)

            if terminated:
                break

        list_loss = []
        list_q_estimate = []
        for step in range(NUM_OPTIM_STEPS):
            (
                batch_state,
                batch_action,
                batch_reward,
                batch_next_state,
                batch_terminated,
            ) = buffer.sample(BATCH_SIZE)

            q_estimate = policy_model({"x": batch_state})["x"].gather(
                dim=-1, index=batch_action
            )

            with torch.no_grad():
                # double DQN, use policy model to select action
                # but use target model to get q value
                # batch_next_action = policy_model({"x": batch_next_state})["x"].argmax(
                #     dim=-1, keepdim=True
                # )

                batch_next_action = target_model({"x": batch_next_state})["x"].argmax(
                    dim=-1, keepdim=True
                )

                q_target = (
                    batch_reward
                    + GAMMA
                    * target_model({"x": batch_next_state})["x"].gather(
                        dim=-1, index=batch_next_action
                    )
                    * ~batch_terminated
                )

            loss = F.smooth_l1_loss(q_estimate, q_target)
            # loss = F.mse_loss(q_estimate, q_target)

            # update NN
            optimizer.zero_grad()
            loss.backward()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=10.0)
            optimizer.step()

            list_loss.append(loss.item())
            list_q_estimate.append(q_estimate.mean().item())

            if step % 50 == 0:
                writer.add_scalar(f"Episode[{episode}]/Loss", loss.item(), step)
                writer.add_scalar(
                    f"Episode[{episode}]/Q", q_estimate.mean().item(), step
                )

        model_update(target_model, policy_model, tau=1.0)

        episode_length = len(list_reward)
        total_reward = sum(list_reward)
        mean_q = sum(list_q_estimate) / len(list_q_estimate)
        mean_loss = sum(list_loss) / len(list_loss)

        writer.add_scalar("Episode/Reward", total_reward, episode)
        writer.add_scalar("Episode/MeanQ", mean_q, episode)
        writer.add_scalar("Episode/AvgLoss", mean_loss, episode)
        writer.add_scalar("Episode/Length", episode_length, episode)


if __name__ == "__main__":
    main()