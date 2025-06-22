import argparse
from pathlib import Path
import  json
from Tokenizer.BPE_Tokenizer_Optimized import train_bpe
from tests.common import gpt2_bytes_to_unicode


def save_bpe(output_path: Path, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
    byte_encoder = gpt2_bytes_to_unicode()
    vocab_json = {}

    for token_id, token_bytes in vocab.items():
        token_str = ''.join([byte_encoder[b] for b in token_bytes])
        vocab_json[token_str] = token_id

    with open(output_path / "train_bpe_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    with open(output_path / "train_bpe_merges.txt", "w", encoding="utf-8") as f:
        for left_bytes, right_bytes in merges:
            left = ''.join([byte_encoder[b] for b in left_bytes])
            right = ''.join([byte_encoder[b] for b in right_bytes])
            f.write(f"{left} {right}\n")

if __name__ == "__main__":
    import pstats
    import cProfile

    parse = argparse.ArgumentParser(description="Training script for BPE Tokenizer")
    parse.add_argument("--path", type=str, required=True,  help="Path to base directory")
    parse.add_argument("--vocab_size", type=int, required=True, help="vocabulary size")
    parse.add_argument("--output_path", type=str, required=True, help="output_path")

    args = parse.parse_args()
    path = Path(args.path)

    output_path = Path(args.output_path)
    Path.mkdir(output_path, parents=True, exist_ok=True)
    special_tokens = ["<|endoftext|>"]

    with cProfile.Profile() as pr:
        _vocab, _merges = train_bpe(path, args.vocab_size, special_tokens)

    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(25)

    save_bpe(output_path, _vocab, _merges)