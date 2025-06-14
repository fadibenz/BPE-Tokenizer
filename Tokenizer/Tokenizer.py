import json
from collections import defaultdict
from typing import Iterable, Iterator
from Tokenizer.BPE_Tokenizer_Optimized import pre_tokenization
from pathlib import Path

class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None


def build_linked_list(list_bytes: list[bytes]):
    len_bytes = len(list_bytes)
    list_nodes = [Node(value) for value in list_bytes ]
    len_list_nodes = len(list_nodes)

    for index in range(len_list_nodes):
        if index < len_bytes - 1:
            list_nodes[index].next = list_nodes[index + 1]
        if index > 0:
            list_nodes[index].prev = list_nodes[index - 1]

    return list_nodes

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

    def encode(self, text:str) -> list[int]:
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
                bigram_pairs = defaultdict(set)
                list_of_bytes = [bytes([b]) for b in encoded_token]
                list_nodes = build_linked_list(list_of_bytes)
                len_nodes = len(list_nodes)

                for _id in range(len_nodes - 1):
                    bigram = (list_nodes[_id].value, list_nodes[_id + 1].value)
                    bigram_pairs[bigram].add(list_nodes[_id])


                while True:
                    potential_merges = {pair: rank for pair, rank in self.merge_priority.items()
                                        if pair in bigram_pairs.keys()}

                    if len(potential_merges) == 0:
                        break

                    next_merge = min(potential_merges, key=potential_merges.get)
                    A, B = next_merge
                    C = A + B
                    nodes = bigram_pairs.pop(next_merge)

                    for node in list(nodes):

                        if node.next is None or node.value != A or node.next.value != B:
                            continue

                        node_next = node.next
                        node_next_next = node_next.next

                        # Update Bigrams
                        if node.prev is not None:
                            bigram_pairs[(node.prev.value, A)].discard(node.prev)

                        if node.next is not None:
                            bigram_pairs[(B, node.next.value)].discard(node)

                        # Change Linked list
                        node.value = C
                        node.next = node_next_next

                        if node_next_next is not None:
                            node_next_next.prev = node

                        if node.prev is not None:
                            bigram_pairs[(node.prev.value, C)].add(node.prev)
                        if node.next is not None:
                            bigram_pairs[(C, node.next.value)].add(node)

                        # Cleanup of removed node
                        node_next.next = None
                        node_next.prev = None

                head_node = list_nodes[0]
                while head_node is not None:
                    try:
                        token_ids_list.append(self.reverse_vocab[head_node.value])
                        head_node = head_node.next
                    except Exception:
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
