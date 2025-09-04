import argparse
import json

import yaml

from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    train(args)


def load_json(settings_path):
    if settings_path.endswith(".yaml") or settings_path.endswith(".yml"):
        with open(settings_path, "r") as f:
            param = yaml.safe_load(f)
    elif settings_path.endswith(".json"):
        with open(settings_path) as data_file:
            param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce of multiple continual learning algorthms."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./exps/finetune.json",
        help="Json file of settings.",
    )
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--device", type=str, default="0")

    # # optim
    # parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'])
    return parser


if __name__ == "__main__":
    main()
