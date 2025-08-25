import numpy as np
import matplotlib.pyplot as plt
from Experiments.utils import log_stats
from pathlib import Path

def word_fragmentation_stats(tokenizer, text_samples, path, corpus_name="TinyStories"):
    word_split_counts = []

    for sample in text_samples:
        words = sample.strip().split()
        for word in words:
            if not word:
                continue
            try:
                tokenized = tokenizer.encode(word)
                word_split_counts.append(len(tokenized))
            except Exception as e:
                print(f"Error tokenizing word: {e}")
                continue

    word_split_counts = np.array(word_split_counts)
    total = len(word_split_counts)

    if total == 0:
        print(f"No valid words found for {corpus_name}")
        return

    stats = {
        "word_fragmentation": {
            "mean_tokens_per_word": float(word_split_counts.mean()),
            "one_token": float(np.sum(word_split_counts == 1) / total),
            "two_token": float(np.sum(word_split_counts == 2) / total),
            "three_or_more_tokens": float(np.sum(word_split_counts >= 3) / total),
        }
    }
    output_path = path / f"word_fragmentation_experiment/{corpus_name}"
    Path.mkdir(output_path, parents=True, exist_ok=True)
    log_stats(stats, output_path)

    print(f"\nWord Fragmentation ({corpus_name}):")
    print(f"Mean Tokens per Word: {stats['word_fragmentation']['mean_tokens_per_word']:.2f}")
    print(f"1 Token: {stats['word_fragmentation']['one_token']:.2%}")
    print(f"2 Tokens: {stats['word_fragmentation']['two_token']:.2%}")
    print(f"≥3 Tokens: {stats['word_fragmentation']['three_or_more_tokens']:.2%}")

    plt.figure()
    bins = np.arange(1, max(word_split_counts.max(), 6) + 1)
    plt.hist(word_split_counts, bins=bins, color="skyblue", edgecolor="black", alpha=0.7, label="BPE")
    plt.title(f"Word Fragmentation Histogram ({corpus_name})")
    plt.xlabel("Tokens per Word")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "word_fragmentation_histogram.png")
    plt.close()