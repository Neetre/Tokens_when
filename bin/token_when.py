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


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BytePairTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.pattern = GPT2_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size):
        assert vocab_size >= 256, "Vocab size must be 256"
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)

        ids = [list(tx.encode("utf-8")) for tx in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = {}

            for chunk_ids in ids:
                get_stats(chunk_ids, stats)

            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, top_pair, idx) for chunk_ids in ids]
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            print(f"Merge {i+1}/{num_merges}: {top_pair} --> {idx}  | {vocab[idx]} had {stats[top_pair]} occurencies!!")

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        part_bytes = []

        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token {idx}")

        tokens = b"".join(part_bytes)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def encode_ordinary(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)

        return ids
    
    def save_merges(self):
        with open("../data/merges.json", "w") as file:
            db ={}
            mer = {str(k): v for k, v in self.merges.items()}
            vocab = {str(k): v.decode("utf-8", errors="replace") for k, v in self.vocab.items()}
            db["merges"] = mer
            db["vocab"] = vocab
            json.dump(db, file, indent=4)

    def load_merges(self):
        with open("../data/merges.json", "r") as file:
            db = json.load(file)
            merges = {eval(k): v for k, v in db["merges"].items()}
            vocab = {eval(k): v.encode("utf-8") for k, v in db["vocab"].items()}
            self.merges = merges
            self.vocab = vocab
    

def get_data():
    with open("../data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return text

def main():
    text = get_data()
    tokenizer = BytePairTokenizer()
    tokenizer.load_merges()
    print("Loaded merges: ", len(tokenizer.merges))
    print("Loaded vocab: ", len(tokenizer.vocab))
    # tokenizer.train(text, 306)
    ids = tokenizer.encode_ordinary(text)
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
