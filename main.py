import click
import json
import utils.nn
from utils.nn.module import setup_module


@click.command()
@click.option("--config_path", type=click.Path(exists=True), required=True)
def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
    model = setup_module(config["model"])


if __name__ == "__main__":
    main()
