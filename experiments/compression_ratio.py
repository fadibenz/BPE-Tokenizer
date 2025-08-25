import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from Experiments.utils import log_stats
from pathlib import Path

def compression_ratio(tokenizer, text_samples, path, corpus_name="TinyStories"):

    compression_ratios = []
    byte_lengths = []
    token_counts = []
    token_ids = []
    byte_level_ratios = []

    for sample in text_samples:
        try:
            bytes_i = len(sample.encode('utf-8', errors='ignore'))
            if bytes_i == 0:
                continue

            tokens_i = tokenizer.encode(sample)
            token_count = len(tokens_i)
            if token_count == 0:
                continue
            compression_ratio_i = bytes_i / token_count
            compression_ratios.append(compression_ratio_i)
            byte_lengths.append(bytes_i)
            token_counts.append(token_count)
            token_ids.append(tokens_i)

            byte_level_ratios.append(1.0)

        except Exception as e:
            print(f"Error processing story: {e}")
            continue

    compression_ratios = np.array(compression_ratios)
    output_path =  path / f"compression_ratio_experiment/{corpus_name}"
    Path.mkdir(output_path, parents=True, exist_ok=True)

    # Statistics
    stats = {
        "compression_ratio": {
            "bpe_mean": float(compression_ratios.mean()),
            "bpe_std": float(compression_ratios.std()),
            "bpe_min": float(compression_ratios.min()),
            "bpe_max": float(compression_ratios.max()),
            "byte_level_mean": 1.0
        }
    }
    log_stats(stats, output_path)

    print(f"\nCompression Ratio ({corpus_name}):")
    print(f"BPE Mean: {stats['compression_ratio']['bpe_mean']:.2f}, Std: {stats['compression_ratio']['bpe_std']:.2f}")
    print(f"Byte-Level Mean: 1.00")

    # Histogram
    plt.figure()
    bins = np.histogram_bin_edges(compression_ratios, bins="auto")
    plt.hist(compression_ratios, bins=bins, color="skyblue", edgecolor="black", alpha=0.7, label="BPE")

    plt.axvline(1.0, color="gray", linestyle="--", label="Byte-Level")
    plt.title(f"Compression Ratio Histogram ({corpus_name})")
    plt.xlabel("Bytes per Token")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "compression_ratio_hist.png")
    plt.close()

    # Correlation
    if len(token_counts) > 1:
        corr, p_value = pearsonr(token_counts, compression_ratios)
        plt.figure()
        plt.scatter(token_counts, compression_ratios, alpha=0.6, s=10)
        plt.title(f"Compression Ratio vs Token Count ({corpus_name})\nPearson r={corr:.2f}, p={p_value:.3f}")
        plt.xlabel("Token Count")
        plt.ylabel("Bytes per Token")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / "compression_vs_token_count.png")
        plt.close()

    return compression_ratios, byte_lengths, token_counts, token_ids