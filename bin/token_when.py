'''

https://github.com/openai/tiktoken
https://en.wikipedia.org/wiki/Byte_pair_encoding

Neetre 2024
'''

import regex as re
import json


def get_stats(ids: list, counts=None):
    """
    Get the frequency of each pair of ids in a list of ids.

    Args:
        ids (list): List of ids.
        counts (dict, optional): Dictionary of counts. Defaults to None.

    Returns:
        dict: Dictionary of counts.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    Merge a pair of ids in a list of ids.

    Args:
        ids (list): List of ids.
        pair (): Pair of ids to merge.
        idx (int): Index to replace the pair with.

    Returns:
        list: New list of ids.
    """

    newids = []  # new list of ids
    i = 0
    while i < len(ids):
        #  if not at the very last position AND the pair matches, replace it
        if i < len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class BaseTokenizer:

    def __init__(self) -> None:
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_zise):
        pass

    def encode(self, text):
        pass

    def decode(self, ids):
        pass

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab


class BytePairTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()

    def train(self, text, vocab_zise):
        num_merges = vocab_zise - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = get_stats(ids)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, top_pair, idx)
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            print(f"Merge {i+1}/{num_merges}: {top_pair} --> {idx}  | {vocab[idx]} had {stats[top_pair]} occurencies!!")

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        tokens = b"".join([self.vocab[idx] for idx in ids])
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def save_merges(self):
        with open("../data/merges.json", "w") as file:
            json.dump(self.merges, file, indent=4)

    def load_merges(self):
        with open("../data/merges.json", "r") as file:
            self.merges = json.load(file)
    

def get_data():
    with open("../data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return text

def main():
    text = get_data()
    tokenizer = BytePairTokenizer()
    tokenizer.train(text, 306)
    ids = tokenizer.encode(text)
    text_de = tokenizer.decode(ids)

    print("---")
    print("Text Lenght: ", len(text))
    print("Tokens length: ", len(ids))
    print("Original text == Decoded text? ", text_de == text)
    print("Vocab size: ", len(tokenizer.vocab))
    print(f"Compression ratio: {len(text) / len(ids):.2f}X\n")

    tokenizer.save_merges()


if __name__ == "__main__":
    main()
