from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
from Tokenizer.Tokenizer import Tokenizer

def top_k_longest_tokens(tokenizer: Tokenizer, token_ids, path, k=50, corpus_name="TinyStories"):
    token_freqs = Counter([t for sample in token_ids for t in sample])
    token_lengths = [
        (token_id, token_bytes.decode("utf-8", errors="replace"), len(token_bytes), token_freqs.get(token_id, 0))
        for token_id, token_bytes in tokenizer.vocab.items()
    ]
    sorted_tokens = sorted(token_lengths, key=lambda x: -x[2])[:k]

    output_path = path / f"top_k_tokens/{corpus_name}"
    Path.mkdir(output_path, parents=True, exist_ok=True)

    print(f"\nTop {k} Longest Tokens ({corpus_name}):")
    for token_id, token_str, length, freq in sorted_tokens:
        print(f"ID {token_id:5d} | Length: {length:3d} | Freq: {freq:5d} | Token: {repr(token_str)}")

    # Visualize top-10
    top_10 = sorted_tokens[:10]
    tokens = [t[1] for t in top_10]
    lengths = [t[2] for t in top_10]
    plt.figure(figsize=(10, 5))
    plt.bar(tokens, lengths, color="skyblue", edgecolor="black")
    plt.title(f"Top-10 Longest Tokens ({corpus_name})")
    plt.xlabel("Token")
    plt.ylabel("Length (Characters)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "top_10_tokens.png")
    plt.close()

    return sorted_tokens