import argparse
import time
from pathlib import Path
import numpy as np
import tempfile

from Experiments.utils import log_stats
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
import os


def process_chunk_tokens(file_path, output_path, tokenizer , chunk_size = 250_000_000):

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        temp_files = []
        buffer_tokens = np.empty(chunk_size, dtype=np.uint16)
        buffer_idx = 0
        size = 0
        with (open(file_path, "r", encoding="utf-8") as f):
            for _id in tokenizer.encode_iterable(f):
                buffer_tokens[buffer_idx] = _id
                buffer_idx += 1
                if buffer_idx == chunk_size:
                    temp_file = temp_dir / f"temp_file_{len(temp_files)}.npy"
                    temp_files.append(temp_file)
                    np.save(temp_file, buffer_tokens)
                    size += chunk_size
                    buffer_idx = 0
        if buffer_idx > 0:
            temp_file = temp_dir / f"temp_file_{len(temp_files)}.npy"
            temp_files.append(temp_file)
            np.save(temp_file, buffer_tokens[:buffer_idx])
            size += buffer_idx

        final_array = np.lib.format.open_memmap(
            output_path,
            "w+",
            np.uint16,
            (size, )
        )
        current_pos = 0
        for file in temp_files:
            chunk = np.load(file)
            len_chunk = len(chunk)
            final_array[current_pos: current_pos + len_chunk] = chunk
            current_pos += len_chunk

if __name__ == "__main__":
    import pstats
    import cProfile

    parser = argparse.ArgumentParser(description="Script for running tokenization")

    parser.add_argument("--tokenizer_path", type=str,required=True , help="Path for directory that contains merges and vocab")
    parser.add_argument("--eval_path", type=str,required=True , help="Path for directory that contains merges and vocab")

    args = parser.parse_args()
    path_tokenizer = Path(args.tokenizer_path)

    special_tokens = ["<|endoftext|>"]
    tokenizer = get_tokenizer_from_vocab_merges_path(path_tokenizer / "train_bpe_vocab.json", path_tokenizer / "train_bpe_merges.txt", special_tokens)
    token_id_list = []

    file_path = Path(args.eval_path)
    file_size_bytes = os.path.getsize(file_path)


    output_path = path_tokenizer / "tokenized_test.npy"

    print("Starting Tokenization")
    start_time = time.time()

    with cProfile.Profile() as pr:
        process_chunk_tokens(file_path, output_path, tokenizer)
    end_time = time.time()
    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(25)

    # Compute and print throughput
    elapsed = end_time - start_time
    throughput_bytes = file_size_bytes / elapsed

    stats = {
        "elapsed_time": elapsed,
        "throughput":throughput_bytes
    }

    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"File size: {file_size_bytes / 1_000_000:.2f} MB")
    print(f"Throughput: {throughput_bytes / 1_000_000:.2f} MB/sec ({throughput_bytes:.2f} bytes/sec)")
    