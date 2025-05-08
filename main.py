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

from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
import os
import time

log_dir = os.path.join("runs", "dqn_" + time.strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir)

# import os
# os.environ["SDL_AUDIODRIVER"] = "dummy"


class ExponentialMovingAverage:
    def __init__(self, alpha=0.995):
        self.alpha = alpha
        self.store = defaultdict(lambda: 0.0)

    def update(self, key, value):
        self.store[key] = self.alpha * self.store[key] + (1 - self.alpha) * value

    def get(self, key):
        return self.store[key]


@click.command()
@click.option("--model_config", type=click.Path(exists=True), required=True)
def main(model_config: str):
    with open(model_config, "r") as f:
        model_config = json.load(f)
    policy_model = setup_module(model_config).to("mps")
    target_model = copy.deepcopy(policy_model).eval()

    def soft_update(target_model, source_model, tau=0.05):
        for target_param, source_param in zip(
            target_model.parameters(), source_model.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),  # [H, W, C] -> [C, H, W]
            transforms.Grayscale(num_output_channels=1),  # [C, H, W] -> [1, H, W]
            transforms.Resize((84, 84)),  # [1, H, W] -> [1, 84, 84]
            transforms.Lambda(
                lambda x: x.unsqueeze(0)
            ),  # [1, 84, 84] -> [1, 1, 84, 84]
            transforms.Lambda(
                lambda x: x.to("mps")
            ),  # [1, 1, 84, 84] -> [1, 1, 84, 84]
        ]
    )
    print(preprocess)

    # env = gym.make("MsPacman-v4", render_mode="human")
    env = gym.make("MsPacman-v4")

    RENDER_FREQ = 0
    SLEEP_TIME = 0.0
    NUM_EPISODES = 1000
    MAX_NUM_STEPS = 1000
    EPSILON = 0.2
    GAMMA = 0.999
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=0.003)

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        list_reward = []
        list_q_est = []
        list_loss = []
        for step in range(MAX_NUM_STEPS):
            if RENDER_FREQ > 0 and episode % RENDER_FREQ == 0:
                env.render()

            # compute current Q
            q_curr = policy_model({"x": preprocess(obs)})["x"]  # [B, C, H, W] -> [B, A]

            # epsilon-greedy sampling & compute next Q
            if random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = q_curr.argmax(dim=-1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            q_next = target_model({"x": preprocess(obs)})["x"]

            # compute loss
            q_est = q_curr[:, action]
            q_tgt = reward + GAMMA * q_next.max(dim=-1)[0] * (1 - terminated)
            loss = F.mse_loss(q_est, q_tgt)

            # update NN
            optimizer.zero_grad()
            loss.backward()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=10.0)
            optimizer.step()

            soft_update(target_model, policy_model, tau=0.05)

            list_reward.append(reward)
            list_q_est.append(q_est.item())
            list_loss.append(loss.item())

            if terminated:
                break

            if SLEEP_TIME > 0:
                sleep(SLEEP_TIME)

        writer.add_scalar("Episode/Reward", sum([GAMMA**i * reward for i, reward in enumerate(list_reward)]), episode)
        writer.add_scalar("Episode/MeanQ", sum(list_q_est) / len(list_q_est), episode)
        writer.add_scalar("Episode/AvgLoss", sum(list_loss) / len(list_loss), episode)


if __name__ == "__main__":
    main()
