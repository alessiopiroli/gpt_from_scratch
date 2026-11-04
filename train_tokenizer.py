import argparse

from gpt.tokenizer.tokenizer import Tokenizer
from gpt.utils.misc import load_config


def main(args):
    cfg = load_config(args.cfg)
    tokenizer = Tokenizer(cfg, cfg.TOKENIZER.tokenizer_input)
    tokenizer.create_vocab_and_merges()
    tokenizer.tokenize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, default="gpt/config/gpt_config.yaml", help="cfg path")
    args = parser.parse_args()
    main(args)
