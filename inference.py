import argparse

from gpt.inferer.inferer import Inferer
from gpt.utils.misc import load_config


def main(args):
    cfg = load_config(args.cfg)
    inferer = Inferer(cfg)
    inferer.inference(cfg.INFERENCE.max_new_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, default="gpt/config/gpt_config.yaml")
    args = parser.parse_args()
    main(args)
