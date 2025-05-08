import click
import json
from utils.nn.module import setup_module

from torchvision import transforms

import gym
from time import sleep


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
        ]
    )
    print(preprocess)

    env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human")
    obs, info = env.reset()

    q = model({"x": preprocess(obs).unsqueeze(0)})["x"]
    with torch.no_grad():
        q = model({"x": preprocess(obs).unsqueeze(0)})["x"]

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            obs, info = env.reset()
        sleep(0.01)


if __name__ == "__main__":
    main()
