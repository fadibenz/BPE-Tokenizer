import json
from collections import defaultdict
from typing import Iterable, Iterator
from Tokenizer.BPE_Tokenizer_Optimized import pre_tokenization
from pathlib import Path


class Tokenizer:
    def __init__(self, vocab:dict[int, bytes],
                 merges:list[tuple[bytes, bytes]],
                 special_tokens:list[str] | None=None):

        self.vocab = vocab
        self.merge_priority = {pair: rank for rank, pair in enumerate(merges)}
        self.merges = merges
        if not special_tokens:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens
        self.reverse_vocab = {value: key for key, value in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath:str | Path,
                   merges_filepath:str | Path,
                   special_tokens: list[str] | None = None):

        merges = []
        vocab = {}
        try:
            with open(vocab_filepath, 'r') as f:
                vocab_data = json.load(f)
                for key, value in vocab_data.items():
                    vocab[int(value)] = key.encode("utf-8")
        except FileNotFoundError:
            print(f"Error: The file '{vocab_filepath}' was not found.")
        except Exception as e:
                print(f"An error occurred: {e}")

        try:
            with open(merges_filepath, 'rb') as f:
                for line in f:
                    parts = line.strip().split(b" ")
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
                    elif len(parts) == 1 and parts[0] != b'':
                        print(f"Warning: Line '{line.strip()}' did not have two parts and was skipped.")
        except FileNotFoundError:
                    print(f"Error: The file '{merges_filepath}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pretokenized_data = pre_tokenization(text, self.special_tokens)
        token_ids_list = []

        for pre_token in pretokenized_data:
            if pre_token in self.special_tokens:
                encoded_token = pre_token.encode("utf-8")
                try:
                    token_ids_list.append(self.reverse_vocab[encoded_token])
                except KeyError:
                    raise Exception("Special Token Not in Vocabulary")
            else:
                encoded_token = pre_token.encode("utf-8")
                tokens = [bytes([b]) for b in encoded_token]

                while True:
                    pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

                    merge_ranks = [self.merge_priority.get(p, float('inf')) for p in pairs]

                    if all(rank == float('inf') for rank in merge_ranks):
                        break

                    i = min(range(len(merge_ranks)), key=merge_ranks.__getitem__)
                    new_token = tokens[i] + tokens[i + 1]
                    tokens[i:i + 2] = [new_token]

                for tok in tokens:
                    try:
                        token_ids_list.append(self.reverse_vocab[tok])
                    except KeyError:
                        raise Exception(f"Token {tok} not in vocabulary")

        return token_ids_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            chunk_ids = self.encode(chunk)
            for idd in chunk_ids:
                yield idd


    def decode(self, ids: list[int]) -> str:
        byte_sequence = b""
        for idd in ids:
            try:
                byte = self.vocab[idd]
            except KeyError:
                byte = b""
            byte_sequence += byte
        decoded_text = byte_sequence.decode("utf-8", errors="replace")
        return decoded_text