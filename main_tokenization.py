import argparse
import time
from pathlib import Path
import numpy as np
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
import os


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

    file_path = path / "train.txt"
    file_size_bytes = os.path.getsize(file_path)

    start_time = time.time()
    with cProfile.Profile() as pr:
        with open(file_path, "r", encoding="utf-8") as f:
            for _id in tokenizer.encode_iterable(f):
                token_id_list.append(_id)
    end_time = time.time()

    token_id_list = np.array(token_id_list, dtype=np.uint16)
    np.save(path / "tokenized_train", token_id_list)

    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(25)

    # Compute and print throughput
    elapsed = end_time - start_time
    throughput_bytes = file_size_bytes / elapsed

    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"File size: {file_size_bytes / 1_000_000:.2f} MB")
    print(f"Throughput: {throughput_bytes / 1_000_000:.2f} MB/sec ({throughput_bytes:.2f} bytes/sec)")
    