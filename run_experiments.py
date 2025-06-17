from pathlib import Path
import argparse
import numpy as np
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
from Experiments.compression_ratio import compression_ratio
from Experiments.utils import sample_stories
from Experiments.token_length_histogram import token_length_histogram
from Experiments.word_fragmentation_rate import word_fragmentation_stats
from Experiments.top_k_longest_tokens import top_k_longest_tokens

if __name__ == "__main__":
    np.random.seed(2024)

    parser = argparse.ArgumentParser(description="Tokenizer experiments")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Path to vocab.json and merges.txt")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to TinyStories corpus")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--test_tokenizer", type=bool, default=False, help="Run top-k tokens")

    args = parser.parse_args()

    tokenizer_dir = Path(args.tokenizer_dir)
    eval_path = Path(args.eval_path)

    # Load tokenizer
    special_tokens = ["<|endoftext|>"]
    tokenizer = get_tokenizer_from_vocab_merges_path(
        tokenizer_dir / "train_bpe_vocab.json",
        tokenizer_dir / "train_bpe_merges.txt",
        special_tokens
    )
    output_path = Path("Experiments/Results")
    # In-Domain Experiments
    print("\n=== In-Domain Experiments (TinyStories) ===")
    text_samples = sample_stories(eval_path, args.samples)

    compression_ratios, byte_lengths, token_counts, token_ids = compression_ratio(
        tokenizer, text_samples,output_path , "TinyStories"
    )
    token_length_histogram(tokenizer, token_ids, output_path, "TinyStories")
    word_fragmentation_stats(tokenizer, text_samples, output_path, "TinyStories")
    if args.test_tokenizer:
        top_k_longest_tokens(tokenizer, token_ids, output_path, corpus_name="TinyStories")
