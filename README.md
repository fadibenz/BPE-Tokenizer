# BPE Tokenizer Implementation

This project implements a **byte-level Byte-Pair Encoding (BPE) tokenizer** from scratch, 
including vocabulary construction, merge operations, and a tokenizer interface for encoding and decoding. 
The implementation was developed as part of a course assignment on tokenization and tokenizer training for language models.


> **Important:**
> This project is inspired from CS336: Assignment 1, 2025. All tests are taken form the public
> repository for the course.

---

## Features

* **BPE Training [`tokenizer/training`](./tokenizer/training)**

  * Includes two versions, a naive implementation and an optimized one; the optimized implementation is 10x faster (run `tests/test_train_bpe.py`)
  * Trains a byte-level BPE tokenizer from raw text.
  * Supports:
    * Initial byte vocabulary.
    * Vocabulary growth through iterative merges.
    * Custom `vocab_size`.
    * User-defined `special_tokens`.
  * Returns:
    * `vocab`: mapping from token IDs → bytes.
    * `merges`: ordered list of merge operations.
  
* **Tokenizer ([`Tokenizer`](./tokenizer/tokenization/tokenizer.py) class)**

  * Encodes text into token IDs and decodes IDs back into text.
  * Supports user-defined special tokens.
  * Provides efficient tokenization for large datasets with streaming support.
  * Interfaces:

    ```python
    def __init__(self, vocab, merges, special_tokens=None)
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
    def encode(self, text: str) -> list[int]
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
    def decode(self, ids: list[int]) -> str
    ```

* **Experiments**

  * **TinyStories (10K vocab)**

    * Added `<|endoftext|>` special token.
    * Serialized vocab + merges.
    * Analyzed training time, memory usage, and longest token.
    * Profiled code for bottlenecks.
  * **OpenWebText (32K vocab)**
    * Serialized vocab + merges.
    * Compared longest tokens vs TinyStories.
    * Qualitative and quantitative comparison between the two tokenizers.
  * **Tokenizer Efficiency Experiments**
    * Compression ratio (bytes/token) on TinyStories vs OpenWebText.
    * Cross-tokenization analysis (TinyStories tokenizer on OWT).
    * Throughput estimation (bytes/second, scaling to The Pile 825GB).
    * Different tokenizer statistics.
---

##  Installation & Setup

This project uses **Python 3.10+** and `uv` for dependency management.

```bash
# Clone repo
git clone <repo_url>
cd bpe-tokenizer
# Install dependencies
uv sync
```
---

##  Usage

### 1. Train a BPE tokenizer

```python
from train_bpe import train_bpe

vocab, merges = train_bpe(
    input_path="data/corpus.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"]
)
```

### 2. Save vocab & merges

```python
import json, pickle

with open("vocab.json", "w") as f:
    json.dump({k: v.decode("utf-8", errors="replace") for k,v in vocab.items()}, f)

with open("merges.pkl", "wb") as f:
    pickle.dump(merges, f)
```

### 3. Load tokenizer and encode/decode

```python
from tokenizer import Tokenizer

tok = Tokenizer.from_files("vocab.json", "merges.pkl", special_tokens=["<|endoftext|>"])

ids = tok.encode("Hello world! <|endoftext|>")
print(ids)

text = tok.decode(ids)
print(text)
```

### 4. Streaming encoding

```python
with open("data/large.txt", "r") as f:
    for token_id in tok.encode_iterable(f):
        # process token_id
        pass
```

---

##  Experiments

### TinyStories (10K vocab)

* Training time: `<4 minutes` with multiprocessing.
* Memory: `<16GB`.

### OpenWebText (32K vocab)
* Training time: `~14 hours`.
* Memory: `<16GB`.

---

##  Project Structure

```
bpe-tokenizer/
│── tokenizer              # Core module containing code for training and tokenization
│── tests/                 # Pytest unit tests
│── experiments/           # Scripts for TinyStories/OWT experiments
│── data/                  # Sample datasets
│── README.md              # Project documentation
```

---

##  Testing

Run provided tests with `pytest`:

```bash
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_tokenizer.py
```
