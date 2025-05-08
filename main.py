import click
import json
from utils.nn.module import setup_module

import torch
from torchvision import transforms
from torch.nn import functional as F

import gym
from time import sleep

import random

import os

os.environ["SDL_AUDIODRIVER"] = "dummy"


class ExponentialMovingAverage:
    def __init__(self, alpha=0.999):
        self.alpha = alpha
        self.store = {}

    def update(self, key, value):
        self.store[key] = self.alpha * self.store[key] + (1 - self.alpha) * value

    def get(self, key):
        return self.store[key]


@click.command()
@click.option("--model_config", type=click.Path(exists=True), required=True)
def main(model_config: str):
    with open(model_config, "r") as f:
        model_config = json.load(f)
    model = setup_module(model_config)
    model = model.to("mps")
    print(model)

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

    env = gym.make("MsPacman-v4", render_mode="human")

    RENDER_FREQ = 10
    SLEEP_TIME = 0.0
    NUM_EPISODES = 100
    MAX_NUM_STEPS = 1000
    EPSILON = 0.2
    GAMMA = 0.999
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ema = ExponentialMovingAverage()

    for episode in range(NUM_EPISODES):
        total_reward = 0
        obs_curr, info = env.reset()
        for step in range(MAX_NUM_STEPS):
            if RENDER_FREQ > 0 and episode % RENDER_FREQ == 0:
                env.render()

            q_curr = model({"x": preprocess(obs_curr)})["x"]  # [B, C, H, W] -> [B, A]

            # epsilon-greedy sampling
            if random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = q_curr.argmax(dim=-1).item()
            obs_next, reward, terminated, truncated, info = env.step(action)
            total_reward += GAMMA**step * reward

            with torch.no_grad():
                q_next = model({"x": preprocess(obs_next)})["x"]

            # compute loss
            q_est = q_curr[:, action]
            q_tgt = reward + GAMMA * q_next.max(dim=-1)[0] * (1 - terminated)
            loss = F.mse_loss(q_est, q_tgt)

            ema.update("q_est", q_est.item())
            ema.update("loss", loss.item())
            print(
                f"logging: {q_est.item():.4f} {ema.get('q_est'):.4f}, {loss.item():.4f}, {ema.get('loss'):.4f}"
            )

            # update NN
            optimizer.zero_grad()
            loss.backward()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            obs_curr = obs_next

            if truncated:
                print(obs_next)

            if terminated:
                break

            if SLEEP_TIME > 0:
                sleep(SLEEP_TIME)

        print(f"Episode {episode} finished with reward {total_reward}")


if __name__ == "__main__":
    main()
