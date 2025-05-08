import click
import json
from utils.nn.module import setup_module

import torch
from torchvision import transforms
from torch.nn import functional as F

import gym
from time import sleep

import random

@click.command()
@click.option("--model_config", type=click.Path(exists=True), required=True)
def main(model_config: str):
    with open(model_config, "r") as f:
        model_config = json.load(f)
    model = setup_module(model_config)
    print(model)

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),                       # [H, W, C] -> [C, H, W]
            transforms.Grayscale(num_output_channels=1), # [C, H, W] -> [1, H, W]   
            transforms.Resize((84, 84)),                 # [1, H, W] -> [1, 84, 84]
            transforms.Lambda(lambda x: x.unsqueeze(0)), # [1, 84, 84] -> [1, 1, 84, 84]
        ]
    )
    print(preprocess)

    env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human")

    RENDER = True
    SLEEP_TIME = 0.01
    NUM_EPISODES = 10
    MAX_NUM_STEPS = 1000
    EPSILON = 0.2
    GAMMA = 0.98
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    for episode in range(NUM_EPISODES):
        obs_curr, info = env.reset()
        q_curr = model({"x": preprocess(obs_curr)})["x"] # [B, C, H, W] -> [B, A]
        for step in range(MAX_NUM_STEPS):
            if RENDER:
                env.render()

            # epsilon-greedy sampling
            if random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = q_curr.argmax(dim=-1).item()
            obs_next, reward, terminated, truncated, info = env.step(action)

            q_next = model({"x": preprocess(obs_next)})["x"]
            
            # compute loss
            q_est = q_curr[:, action]
            q_tgt = reward + GAMMA * q_next.max(dim=-1).item() * (1 - terminated)
            loss = F.mse_loss(q_est, q_tgt)

            # update NN
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get ready for the next step
            obs_curr, q_curr = obs_next, q_next

            if SLEEP_TIME > 0:
                sleep(SLEEP_TIME)

            if terminated or truncated:
                break

if __name__ == "__main__":
    main()
