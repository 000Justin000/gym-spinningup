import click
import json
from utils.nn.module import setup_module

# from torchvision import transforms

# import gym
# from time import sleep


@click.command()
@click.option("--model_config", type=click.Path(exists=True), required=True)
def main(config_path: str):
    with open(config_path, "r") as f:
        model_config = json.load(f)
    model = setup_module(model_config)
    print(model)

    # preprocess = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Grayscale(num_output_channels=1),
    #         transforms.Resize((84, 84)),
    #     ]
    # )
    # print(preprocess)

    # env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human")
    # env.reset()
    # env.render()

    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     ob_next, reward, terminated, truncated, info = env.step(action)
    #     env.render()

    #     if terminated or truncated:
    #         env.reset()
    #     sleep(0.01)


if __name__ == "__main__":
    main()
