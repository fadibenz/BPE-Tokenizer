import argparse
from pathlib import Path
import  json
from Tokenizer.BPE_Tokenizer_Optimized import train_bpe
from tests.common import gpt2_bytes_to_unicode


def save_bpe(base_path: Path, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
    byte_encoder = gpt2_bytes_to_unicode()
    vocab_json = {}

    for token_id, token_bytes in vocab.items():
        token_str = ''.join([byte_encoder[b] for b in token_bytes])
        vocab_json[token_str] = token_id

    with open(base_path / "train_bpe_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    with open(base_path / "train_bpe_merges.text", "w", encoding="utf-8") as f:
        for left_bytes, right_bytes in merges:
            left = ''.join([byte_encoder[b] for b in left_bytes])
            right = ''.join([byte_encoder[b] for b in right_bytes])
            f.write(f"{left} {right}\n")

if __name__ == "__main__":
     import pstats
     import cProfile


     parse = argparse.ArgumentParser(description="Training script for BPE Tokenizer")
     parse.add_argument("--path", type=str, required=True,  help="Path to base directory")
     parse.add_argument("--vocabSize", type=int, required=True, help="vocabulary size")
     args = parse.parse_args()
     path = Path(args.path)

     special_tokens = ["<|endoftext|>"]

     with cProfile.Profile() as pr:
        vocab, merges = train_bpe(path / "train.txt", args.vocabSize, special_tokens)

     stats = pstats.Stats(pr)
     stats.sort_stats("cumtime").print_stats(25)

     save_bpe(path, vocab, merges)