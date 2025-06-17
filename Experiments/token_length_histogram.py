import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from Experiments.utils import log_stats


def token_length_histogram(tokenizer, token_ids, path, corpus_name="TinyStories"):
    flat_tokens = [t for sample in token_ids for t in sample]
    token_texts = [tokenizer.vocab.get(t, b"").decode("utf-8", errors="replace") for t in flat_tokens]
    lengths = np.array([len(t) for t in token_texts if len(t) > 0])

    if len(lengths) == 0:
        print(f"No valid tokens found for {corpus_name}")
        return

    output_path = path  / f"token_length_experiment/{corpus_name}"
    Path.mkdir(output_path, parents=True, exist_ok=True)

    # Statistics
    stats = {
        "token_length": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "std": float(lengths.std()),
            "min": float(lengths.min()),
            "max": float(lengths.max())
        }
    }
    log_stats(stats, output_path)

    print(f"\nToken Length ({corpus_name}):")
    print(
        f"Mean: {stats['token_length']['mean']:.2f}, Median: {stats['token_length']['median']:.2f}, Std: {stats['token_length']['std']:.2f}")

    # Histogram
    plt.figure()
    bins = np.histogram_bin_edges(lengths, bins="auto")
    plt.hist(lengths, bins=bins, color="skyblue", edgecolor="black")
    plt.title(f"Token Length Histogram ({corpus_name})")
    plt.xlabel("Token Length (Characters)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "token_length_histogram.png")
    plt.close()