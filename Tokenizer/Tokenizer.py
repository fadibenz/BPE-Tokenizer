import json
from typing import Iterable, Iterator
from Tokenizer.BPE_Tokenizer_Optimized import pre_tokenization
from pathlib import Path
import heapq

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None
        self.alive = True

def build_linked_list(list_bytes: list[bytes]):
    len_bytes = len(list_bytes)
    list_nodes = [Node(value) for value in list_bytes]
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

        import itertools
        heap_counter = itertools.count()

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
                node_list = build_linked_list(tokens)
                heap = []

                for i in range(len(tokens) - 1):
                    pair = (node_list[i].value, node_list[i + 1].value)
                    if pair in self.merge_priority:
                        rank = self.merge_priority[pair]
                        heapq.heappush(heap, (rank, next(heap_counter),  node_list[i], pair))
                while heap:
                    rank, _, node, pair = heapq.heappop(heap)

                    if node.alive == False or node.next is None or (node.value, node.next.value) != pair:
                        continue

                    node.value = pair[0] + pair[1]
                    to_remove = node.next
                    next_next = to_remove.next

                    node.next = next_next

                    if next_next is not None:
                        next_next.prev = node

                    # cleanup
                    to_remove.alive = False
                    to_remove.next = None
                    to_remove.prev = None

                    if node.next is not None and node.next.alive :
                        new_token = (node.value , node.next.value)
                        if new_token in self.merge_priority:
                            heapq.heappush(heap, (self.merge_priority[new_token], next(heap_counter), node, new_token))

                    if node.prev is not None and node.prev.alive:
                        new_token =  (node.prev.value, node.value)
                        if new_token in self.merge_priority:
                            heapq.heappush(heap, (self.merge_priority[new_token], next(heap_counter), node.prev, new_token))

                head = node_list[0]
                while head and not head.alive:
                    head = head.next
                while head:
                    token_ids_list.append(self.reverse_vocab[head.value])
                    head = head.next
        return token_ids_list

    # def encode(self, text: str) -> list[int]:
    #     pretokenized_data = pre_tokenization(text, self.special_tokens)
    #     token_ids_list = []
    #
    #     for pre_token in pretokenized_data:
    #         if pre_token in self.special_tokens:
    #             encoded_token = pre_token.encode("utf-8")
    #             try:
    #                 token_ids_list.append(self.reverse_vocab[encoded_token])
    #             except KeyError:
    #                 raise Exception("Special Token Not in Vocabulary")
    #         else:
    #             encoded_token = pre_token.encode("utf-8")
    #             tokens = [bytes([b]) for b in encoded_token]
    #
    #             while True:
    #                 pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    #                 merge_ranks = [self.merge_priority.get(p, float('inf')) for p in pairs]
    #
    #                 if all(rank == float('inf') for rank in merge_ranks):
    #                     break
    #
    #                 i = min(range(len(merge_ranks)), key=merge_ranks.__getitem__)
    #                 tokens[i:i + 2] = [tokens[i] + tokens[i + 1]]
    #
    #             for tok in tokens:
    #                 try:
    #                     token_ids_list.append(self.reverse_vocab[tok])
    #                 except KeyError:
    #                     raise Exception(f"Token {tok} not in vocabulary")
    #
    #     return token_ids_list

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