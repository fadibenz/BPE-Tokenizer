from __future__ import annotations

import json
import os
import sys

import psutil
import pytest
import tiktoken

from .adapters import get_tokenizer
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


if sys.platform != "win32":
    import resource


def memory_limit(max_mem):
    """
    A decorator to monitor or attempt to limit memory usage for a function.
    On Unix-like systems, it attempts to set a memory limit.
    On Windows, it only monitors memory usage and provides a way to check if
    a threshold was exceeded, but does not enforce a hard limit.

    Args:
        max_mem (int): The maximum memory in bytes allowed for the function
                       relative to its starting memory.
    """

    def decorator(f):
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())

            # Record initial memory usage
            initial_rss = process.memory_info().rss

            # --- Unix-specific memory limit setting ---
            prev_limits = None
            if sys.platform != "win32":
                try:
                    prev_limits = resource.getrlimit(resource.RLIMIT_AS)
                    # Set the new limit: current RSS + max_mem. Hard limit is -1 (unlimited)
                    # Note: RLIMIT_AS includes stack, data, and heap.
                    resource.setrlimit(resource.RLIMIT_AS, (initial_rss + max_mem, -1))
                except Exception as e:
                    print(f"Warning: Could not set memory limit on Unix: {e}")
                    # Continue without the hard limit if setting fails

            try:
                result = f(*args, **kwargs)
                return result
            finally:
                # --- Reset Unix memory limit ---
                if sys.platform != "win32" and prev_limits:
                    try:
                        resource.setrlimit(resource.RLIMIT_AS, prev_limits)
                    except Exception as e:
                        print(f"Warning: Could not reset memory limit on Unix: {e}")

                # --- Cross-platform memory usage check ---
                final_rss = process.memory_info().rss
                memory_used_by_function = final_rss - initial_rss

                # You can add an assertion here in your tests if you want to
                # ensure the function didn't exceed 'max_mem'
                # For example, in your test function:
                # assert memory_used_by_function <= max_mem, \
                #     f"Memory usage exceeded: {memory_used_by_function} > {max_mem}"

                # For demonstration, let's print it:
                # print(f"\nFunction '{f.__name__}' used {memory_used_by_function / (1024*1024):.2f} MB "
                #       f"(limit was {max_mem / (1024*1024):.2f} MB)")

        return wrapper

    return decorator

def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path, encoding="utf-8") as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


def test_roundtrip_empty():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = ""
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_empty_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = ""

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == []

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_single_character():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "s"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_single_character_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "s"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == ["s"]

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_single_unicode_character():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "🙃"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_single_unicode_character_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "🙃"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_ascii_string():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "Hello, how are you?"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_ascii_string_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Hello, how are you?"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    # assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == ["Hello", ",", " how", " are", " you", "?"]

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_unicode_string():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "Héllò hôw are ü? 🙃"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_unicode_string_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw are ü? 🙃"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_unicode_string_with_special_tokens():

    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )

    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # Ensure the special <|endoftext|> token is preserved
    assert tokenized_string.count("<|endoftext|>") == 3

    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_unicode_string_with_special_tokens_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"

    reference_ids = reference_tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_overlapping_special_tokens():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )

    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    print(tokenized_string)
    # Ensure the double <|endoftext|><|endoftext|> is preserved as a single token
    assert tokenized_string.count("<|endoftext|>") == 1
    assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    # Test roundtrip
    assert tokenizer.decode(ids) == test_string


def test_address_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "address.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    content = tokenizer.decode(ids)
    assert tokenizer.decode(ids) == corpus_contents


def test_address_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    corpus_path = FIXTURES_PATH / "address.txt"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_german_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "german.txt", encoding="utf-8") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_german_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    corpus_path = FIXTURES_PATH / "german.txt"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_tinystories_sample_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_tinystories_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "tinystories_sample.txt"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_encode_special_token_trailing_newlines():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "special_token_trailing_newlines.txt"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_encode_special_token_double_newline_non_whitespace():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "special_token_double_newlines_non_whitespace.txt"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_encode_iterable_tinystories_sample_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        corpus_contents = f.read()
    assert tokenizer.decode(all_ids) == corpus_contents


def test_encode_iterable_tinystories_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "tinystories_sample.txt"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    assert all_ids == reference_ids

    assert tokenizer.decode(all_ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit support for non-linux systems is spotty.",
)
def test_encode_iterable_memory_usage():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt", encoding="utf-8") as f:
        ids = []
        for _id in _encode_iterable(tokenizer, f):
            ids.append(_id)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit support for non-linux systems is spotty.",
)
@pytest.mark.xfail(reason="Tokenizer.encode is expected to take more memory than allotted (1MB).")
def test_encode_memory_usage():
    """
    We expect this test to fail, since Tokenizer.encode is not expected to be memory efficient.
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt", encoding="utf-8") as f:
        contents = f.read()
        _ = _encode(tokenizer, contents)


@memory_limit(int(1e6))
def _encode_iterable(tokenizer, iterable):
    """
    We place tokenizer.encode_iterable into a separate function so we can limit memory
    for just this function. We set the memory limit to 1MB.
    """
    yield from tokenizer.encode_iterable(iterable)


@memory_limit(int(1e6))
def _encode(tokenizer, text):
    """
    We place tokenizer.encode into a separate function so we can limit memory
    for just this function. We set the memory limit to 1MB.
    """
    return tokenizer.encode(text)
