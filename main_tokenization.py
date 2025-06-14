import argparse
from pathlib import Path
from Tokenizer.Tokenizer import Tokenizer
import numpy as np
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path


if __name__ == "__main__":
    import pstats
    import cProfile

    parser = argparse.ArgumentParser(description="Script for running tokenization")

    parser.add_argument("--path", type=str,required=True , help="Path for directory that contains merges and vocab")

    args = parser.parse_args()
    path = Path(args.path)
    special_tokens = ["<|endoftext|>"]
    tokenizer = get_tokenizer_from_vocab_merges_path(path / "train_bpe_vocab.json", path / "train_bpe_merges.txt", special_tokens)
    token_id_list = []

    with cProfile.Profile() as pr:
        with open(path / "corpus.en", "r", encoding="utf-8") as f:
            for _id in tokenizer.encode_iterable(f):
                token_id_list.append(_id)

    token_id_list = np.array(token_id_list, dtype=np.uint16)
    np.save(path / "tokenized_valid", token_id_list)

    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(25)
