import json

from tqdm import tqdm


class Tokenizer:
    def __init__(self, cfg, path):
        self.cfg = cfg
        self.path = path
        self.vocab_size = int(self.cfg.TOKENIZER.vocab_size)
        self.tokenizer_data_file = self.cfg.TOKENIZER.tokenizer_data_file

    def get_text(self, path):
        with open(path, "r") as f:
            text = f.read()

        return text

    @staticmethod
    def get_frequencies(tokens):
        frequencies = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair not in frequencies:
                frequencies[pair] = 1
            else:
                frequencies[pair] += 1

        frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True))

        return frequencies

    @staticmethod
    def update_tokens(tokens, most_frequent):
        new_tokens = []
        i = 0

        while i < len(tokens):
            if (i + 1 < len(tokens)) and ((tokens[i], tokens[i + 1]) == most_frequent):
                new_tokens.append("".join(most_frequent))
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def bpe(self, text, vocab_size):
        tokens = list(text)
        vocabulary = set(tokens)
        merges = []

        for _ in tqdm(range(vocab_size - len(vocabulary)), desc="Creating vocabulary and merges"):
            frequencies = self.get_frequencies(tokens)
            if not frequencies:
                break
            most_frequent = next(iter(frequencies))
            tokens = self.update_tokens(tokens, most_frequent)
            vocabulary.add("".join(most_frequent))
            merges.append(most_frequent)

        return vocabulary, merges

    def save_tokenizer(self, vocabulary, merges):
        tokenizer_data = {"vocabulary": list(vocabulary), "merges": merges}

        with open(self.tokenizer_data_file, "w") as f:
            json.dump(tokenizer_data, f)

    @staticmethod
    def load_tokenizer(path):
        with open(path, "r") as f:
            tokenizer_data = json.load(f)

        vocabulary = tokenizer_data["vocabulary"]
        merges = [tuple(m) for m in tokenizer_data["merges"]]

        return vocabulary, merges

    @staticmethod
    def do_one_merge(text, merge):
        new_text = []
        i = 0
        while i < len(text):
            if (i + 1 < len(text)) and (merge == (text[i], text[i + 1])):
                new_text.append("".join(merge))
                i += 2
            else:
                new_text.append(text[i])
                i += 1

        return new_text

    def tokenize_text(self, text, merges):
        text = list(text)

        for merge in tqdm(merges, desc="Tokenizing text"):
            text = self.do_one_merge(text, merge)

        return text

    def create_vocab_and_merges(self):
        text = self.get_text(self.path)
        vocabulary, merges = self.bpe(text, self.vocab_size)
        self.save_tokenizer(vocabulary, merges)

    def save_tokenized_data(self, tokenized_text, vocabulary, merges):
        tokenized_data = {"tokenized_text": tokenized_text, "vocabulary": vocabulary, "merges": merges}

        with open(self.tokenizer_data_file, "w") as f:
            json.dump(tokenized_data, f)

    def tokenize(self):
        text = self.get_text(self.path)
        vocabulary, merges = self.load_tokenizer(self.tokenizer_data_file)
        tokenized_text = self.tokenize_text(text, merges)
        self.save_tokenized_data(tokenized_text, vocabulary, merges)
