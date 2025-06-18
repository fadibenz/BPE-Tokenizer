from pathlib import Path
from typing import List
from collections import Counter
import matplotlib.pyplot as plt
from Experiments.utils import sample_stories, log_stats
from Experiments.compression_ratio import compression_ratio
from Experiments.word_fragmentation_rate import word_fragmentation_stats
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path


def data_leakage_experiment(
        tokenizer_a,
        tokenizer_b,
        valid_path: Path,
        test_path: Path,
        output_path: Path,
        samples: int = 1000,
):
    output_path.mkdir(parents=True, exist_ok=True)

    # Sample validation and test sets
    valid_samples = sample_stories(valid_path, samples, delimiter="<|endoftext|>")
    test_samples = sample_stories(test_path, samples, delimiter="<|endoftext|>")

    # Compute token overlap
    def compute_token_overlap(tokenizer, samples: List[str]) -> Counter:
        token_counts = Counter()
        for sample in samples:
            tokens = tokenizer.encode(sample)
            token_counts.update(tokens)
        return token_counts


    valid_a_tokens = compute_token_overlap(tokenizer_a, valid_samples)
    test_a_tokens = compute_token_overlap(tokenizer_a, test_samples)
    valid_b_tokens = compute_token_overlap(tokenizer_b, valid_samples)
    test_b_tokens = compute_token_overlap(tokenizer_b, test_samples)

    # Overlap metrics
    valid_a_set = set(valid_a_tokens.keys())
    test_a_set = set(test_a_tokens.keys())
    valid_b_set = set(valid_b_tokens.keys())
    test_b_set = set(test_b_tokens.keys())

    overlap_a = len(valid_a_set & test_a_set) / len(valid_a_set | test_a_set)
    overlap_b = len(valid_b_set & test_b_set) / len(valid_b_set | test_b_set)

    stats = {
        "token_overlap": {
            "tokenizer_a": overlap_a,
            "tokenizer_b": overlap_b
        }
    }

    print("\nToken Overlap (Validation vs. Test):")
    print(f"Tokenizer A: {overlap_a:.2%}")
    print(f"Tokenizer B: {overlap_b:.2%} (higher due to validation leakage)")

    # Compute compression metrics
    print("\nCompression Metrics (Validation):")
    valid_a_stats, _, _, _ = compression_ratio(tokenizer_a, valid_samples, output_path, "TokenizerA_Valid")
    valid_b_stats, _, _, _ = compression_ratio(tokenizer_b, valid_samples, output_path, "TokenizerB_Valid")
    word_fragmentation_stats(tokenizer_a, valid_samples, output_path, "TokenizerA_Valid")
    word_fragmentation_stats(tokenizer_b, valid_samples, output_path, "TokenizerB_Valid")

    print("\nCompression Metrics (Test):")
    test_a_stats, _, _, _ = compression_ratio(tokenizer_a, test_samples, output_path, "TokenizerA_Test")
    test_b_stats, _, _, _ = compression_ratio(tokenizer_b, test_samples, output_path, "TokenizerB_Test")
    word_fragmentation_stats(tokenizer_a, test_samples, output_path, "TokenizerA_Test")
    word_fragmentation_stats(tokenizer_b, test_samples, output_path, "TokenizerB_Test")

    # Update stats
    stats.update({
        "validation_metrics": {
            "tokenizer_a_bytes_per_token": valid_a_stats[0].mean(),
            "tokenizer_b_bytes_per_token": valid_b_stats[0].mean(),
            "tokenizer_a_fragmentation": valid_a_stats[1]["word_fragmentation"]["three_or_more_tokens"],
            "tokenizer_b_fragmentation": valid_b_stats[1]["word_fragmentation"]["three_or_more_tokens"]
        },
        "test_metrics": {
            "tokenizer_a_bytes_per_token": test_a_stats[0].mean(),
            "tokenizer_b_bytes_per_token": test_b_stats[0].mean(),
            "tokenizer_a_fragmentation": test_a_stats[1]["word_fragmentation"]["three_or_more_tokens"],
            "tokenizer_b_fragmentation": test_b_stats[1]["word_fragmentation"]["three_or_more_tokens"]
        }
    })
    log_stats(stats, output_path)

    # Visualize
    plt.figure()
    plt.bar(["Tokenizer A", "Tokenizer B"], [overlap_a, overlap_b], color=["skyblue", "salmon"], edgecolor="black")
    plt.title("Token Overlap (Validation vs. Test)")
    plt.ylabel("Overlap Ratio")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "token_overlap.png")
    plt.close()

    plt.figure()
    metrics = [
        stats["validation_metrics"]["tokenizer_a_bytes_per_token"],
        stats["validation_metrics"]["tokenizer_b_bytes_per_token"],
        stats["test_metrics"]["tokenizer_a_bytes_per_token"],
        stats["test_metrics"]["tokenizer_b_bytes_per_token"]
    ]
    labels = ["A (Valid)", "B (Valid)", "A (Test)", "B (Test)"]
    plt.bar(labels, metrics, color=["skyblue", "salmon", "skyblue", "salmon"], edgecolor="black")
    plt.title("Bytes per Token Comparison")
    plt.ylabel("Bytes per Token")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "bytes_per_token.png")
    plt.close()

    return stats