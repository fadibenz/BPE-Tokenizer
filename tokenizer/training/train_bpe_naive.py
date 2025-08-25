import regex as re
from collections import Counter, defaultdict

def contains_subtuple(big, sub):
    matches = []
    len_big = len(big)

    for i in range(len_big - 2 + 1):
        if big[i:i + 2] == sub:
            matches.append(i)
    return matches

def pre_tokenization(training_data, special_tokens):
    # This is taken form github.com/openai/tiktoken/pull/234/files (GPT2)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # splitting on special tokens
    escaped_list = [re.escape(special_token) for special_token in special_tokens]
    split_PAT = f"({"|".join(escaped_list)})"
    split_corpus = re.split(split_PAT, training_data)
    # Pre-tokenization
    pretokenized_train_data = []
    for doc in split_corpus:
        if doc in special_tokens:
            pretokenized_train_data.append(doc)
        else:
            for m in re.finditer(PAT, doc):
                pretokenized_train_data.append(m.group())
    return pretokenized_train_data

def merging(constructed_vocab, new_merge, token_id):
    to_add = {}
    to_delete = []
    for word in list(constructed_vocab):
        indices = contains_subtuple(word, new_merge)
        if indices:
            key = word
            for i in indices:
                if i + 2 < len(word) - 1:
                    key = key[:i] + (token_id,) + key[i+2:]
                else:
                    key = key[:i] + (token_id,)
            count = constructed_vocab[word]
            to_delete.append(word)
            to_add[key] = count
    for k in to_delete:
        del constructed_vocab[k]
    constructed_vocab.update(to_add)

def find_best_pair(constructed_vocab):
        potential_merges_count = {}
        for index, (word, count) in enumerate(constructed_vocab.items()):
            for i in range(len(word) - 1):
                key = (word[i], word[i+1])
                potential_merges_count[key] = potential_merges_count.get(key, 0) + count

        v = max(potential_merges_count.values())
        keys = [k for k, val in potential_merges_count.items() if val == v]
        new_merge = max(keys)
        return new_merge

def train_bpe(file_path: str,
              vocab_size: int,
              special_tokens: list[str],
              ):
    merges = []
    # reading file
    try:
        with open(file_path, 'r') as f:
            training_data = f.read()
    except Exception as e:
        print(f"error: {e}")
        return

    if not isinstance(special_tokens, list):
        special_tokens = []

    # initial vocabulary
    vocab = {i : bytes([i]) for i in range(256)}
    # Pre-tokenization
    pretokenized_train_data = pre_tokenization(training_data, special_tokens)

    # Constructing the initial frequency dictionary
    ctr = Counter(pretokenized_train_data)
    constructed_vocab = {}
    for word, count in ctr.items():
        if len(word) == 1: continue
        constructed_vocab[tuple(word.encode("utf-8"))] = count

    token_id = (len(vocab) + len(special_tokens)) - 1
    while token_id < vocab_size - 1:
        best_pair = find_best_pair(constructed_vocab)
        merging(constructed_vocab, best_pair, token_id)
        token_id += 1
    return vocab, merges


def main():
    vocab_size = 257
    special_tokens = ['<|endoftext|>']
    train_bpe('./test.txt', vocab_size, special_tokens)

if __name__ == "__main__":
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(25)