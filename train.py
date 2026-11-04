import argparse

from gpt.trainer.trainer import Trainer
from gpt.utils.misc import load_config


def main(args):
    cfg = load_config(args.cfg)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, default="gpt/config/gpt_config.yaml", help="cfg path")
    args = parser.parse_args()
    main(args)
