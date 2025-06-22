import argparse
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
from pathlib import Path
from Experiments.data_leakage_experiment.data_leakage import data_leakage_experiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Running experiment for data leakage")
    parser.add_argument("--tokenizer_a", type=str, required=True,
                        help="Path to vocab.json and merges.txt for tokenizer without leak")
    parser.add_argument("--tokenizer_b", type=str, required=True,
                        help="Path to vocab.json and merges.txt for tokenizer with leak")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to corpus")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")

    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]

    path_corpus = Path(args.eval_path)

    tokenizer_a = get_tokenizer_from_vocab_merges_path(
        Path(args.tokenizer_a) / "train_bpe_vocab.json",
        Path(args.tokenizer_a) / "train_bpe_merges.txt",
        special_tokens=special_tokens
    )

    tokenizer_b = get_tokenizer_from_vocab_merges_path(
        Path(args.tokenizer_b) / "train_bpe_vocab.json",
        Path(args.tokenizer_b) / "train_bpe_merges.txt",
        special_tokens=special_tokens
    )
    print("\n Running data leakage experiment")

    data_leakage_experiment(tokenizer_a, tokenizer_b,
                            path_corpus / "valid.txt", path_corpus / "test.txt",
                            Path("Experiments/Results/data_leakage/OpenWebText"),
                            args.samples)