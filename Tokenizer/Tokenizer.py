import json
from typing import Iterable, Iterator
import regex as re
from cs336_basics.Tokenizer.BPE_Tokenizer_Optimized import pre_tokenization

class Tokenizer:
    def __init__(self, vocab:dict[int, bytes],
                 merges:list[tuple[bytes, bytes]],
                 special_tokens:list[str] | None=None):
        self.vocab = vocab
        self.merges = merges
        if not special_tokens:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens
        self.reverse_vocab = {value: key for key, value in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath:str,
                   merges_filepath:str,
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

    def encode(self, text:str) -> list[int]:
        pretokenized_data = pre_tokenization(text, self.special_tokens)
        token_ids_list = []

        for pre_token in pretokenized_data:
            token_list = [special_token for special_token in self.special_tokens if special_token in pre_token]

            if len(token_list) >=1:
                    escaped_list = [re.escape(special_token) for special_token in token_list]
                    split_PAT = r"({})".format("|".join(escaped_list))
                    token_list = re.findall(split_PAT, pre_token)
                    for token in token_list:
                        encoded_token = token.encode("utf-8")
                        try:
                            token_id = self.reverse_vocab[encoded_token]
                            token_ids_list.append(token_id)
                        except KeyError:
                            raise Exception("Special Token Not in Vocabulary")
            else:
                encoded_bytes = pre_token.encode("utf-8")
                list_of_bytes = [bytes([b]) for b in encoded_bytes]
                for merge in self.merges:

                    if len(list_of_bytes) <= 1:
                        break

                    if len(merge) == 0 or len(merge) == 1 :
                        continue

                    merged = True

                    while merged:
                        merged = False
                        for i in range(len(list_of_bytes) - 1):
                            if (list_of_bytes[i] == merge[0]) and\
                                    (list_of_bytes[i+1] == merge[1]):
                                list_of_bytes = (list_of_bytes[:i] + [merge[0] + merge[1]]
                                                 + list_of_bytes[i+2:])
                                merged = True
                                break

                for token in list_of_bytes:
                    try:
                        token_id = self.reverse_vocab[token]
                        token_ids_list.append(token_id)
                    except StopIteration:
                        raise Exception("Vocabulary is not consistent with merges")
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




if __name__ == "__main__":
    string = "the cat ate<|endoftext|>"
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h',
             5: b't', 6: b'th', 7: b' c', 8: b' a',
             9: b'the', 10: b' at', 11:b'<|endoftext|>'}
    merges = [(b't', b'h'), (b' ', b'c'),
              (b' ', b'a'), (b'th', b'e'),
              (b' a', b't')]
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    print(tokenizer.encode("s"))
