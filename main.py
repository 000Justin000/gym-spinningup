import click
import json
from utils.nn.module import setup_module


@click.command()
@click.option("--model_config", type=click.Path(exists=True), required=True)
def main(config_path: str):
    with open(config_path, "r") as f:
        model_config = json.load(f)
    model = setup_module(model_config)
    print(model)


if __name__ == "__main__":
    main()
