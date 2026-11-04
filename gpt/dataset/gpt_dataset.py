import json

import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = self.cfg.TRAINING.batch_size
        self.context_size = self.cfg.TRAINING.context_size
        self.tokenizer_data_file = self.cfg.TOKENIZER.tokenizer_data_file
        self.toknized_text, self.vocabulary = self.load_tokenized_data()

        self.stoi = {tok: i for i, tok in enumerate(self.vocabulary)}
        self.itos = {i: tok for i, tok in enumerate(self.vocabulary)}

        self.encode = lambda s: [self.stoi[tok] for tok in s]
        self.decode = lambda l: "".join(self.itos[i.item()] for i in l)

        self.data = torch.tensor(self.encode(self.toknized_text), dtype=torch.long)
        self.n = int(0.9 * len(self.data))
        self.data = {"train": self.data[: self.n], "validation": self.data[self.n :]}

    def get_batch(self, split):
        # starting points for training contexts
        ix = torch.randint(0, len(self.data[split]) - self.context_size, (self.batch_size, 1))
        x = torch.stack([self.data[split][i : i + self.context_size] for i in ix])
        y = torch.stack([self.data[split][i + 1 : i + self.context_size + 1] for i in ix])

        return x, y

    def load_tokenized_data(self):
        with open(self.tokenizer_data_file, "r") as f:
            tokenizer_data = json.load(f)

        tokenized_text = tokenizer_data["tokenized_text"]
        vocabulary = tokenizer_data["vocabulary"]

        return tokenized_text, vocabulary
