from pathlib import Path
import random


def story_generator(input_path: Path, delimiter: str = "<|endoftext|>"):
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        buffer = ""
        while chunk := f.read(512 * 512):
            buffer += chunk
            parts = buffer.split(delimiter)
            buffer = parts[-1]  # incomplete part, carry forward
            for part in parts[:-1]:
                if part.strip():
                    yield part.strip()
        if buffer.strip():
            yield buffer.strip()


def split_train_valid(input_path: Path, output_dir: Path, train_size: int, valid_size: int, seed: int = 2024):
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Counting total stories... (this may take a while)")
    total_stories = sum(1 for _ in story_generator(input_path))
    print(f"Total stories found: {total_stories}")

    if train_size + valid_size > total_stories:
        raise ValueError(
            f"Requested train_size ({train_size}) + valid_size ({valid_size}) is greater than total stories ({total_stories})")

    all_indices = list(range(total_stories))
    random.shuffle(all_indices)

    valid_indices = set(all_indices[:valid_size])
    train_indices = set(all_indices[valid_size: valid_size + train_size])

    train_written = 0
    valid_written = 0

    print("Splitting and writing...")
    with open(output_dir / "train.txt", "w", encoding="utf-8") as train_f, \
            open(output_dir / "valid.txt", "w", encoding="utf-8") as valid_f:
        for i, story in enumerate(story_generator(input_path)):
            if i in valid_indices:
                valid_f.write(story + "<|endoftext|>")
                valid_written += 1
            elif i in train_indices:
                train_f.write(story + "<|endoftext|>")
                train_written += 1

            if (i + 1) % 100000 == 0:
                print(f"Processed {i + 1}/{total_stories} stories...")

            if train_written == train_size and valid_written == valid_size:
                print("Collected all required stories. Stopping early.")
                break

    print(f"Saved {valid_written} stories to valid.txt and {train_written} to train.txt")


def append_file_to_file(source_file_path, destination_file_path):

    CHUNK_SIZE = 1024 *  1024 * 8
    with open(source_file_path, "r", encoding="utf-8") as src_f:
        with open(destination_file_path, "a", encoding="utf-8") as dst_f:
            while chunk := src_f.read(CHUNK_SIZE):
                dst_f.write(chunk)

if __name__ == "__main__":
    # source_path = Path("data/TinyStoriesV2-GPT4/amplified_split/valid.txt")
    # dest_path = Path("data/TinyStoriesV2-GPT4/amplified_split/train+valid.txt")
    # append_file_to_file(source_path, dest_path)

    input_path = Path("data/TinyStoriesV2-GPT4/train.txt")
    output_dir = Path("data/TinyStoriesV2-GPT4/amplified_split")

    train_stories = 550000
    valid_stories = 220000

    split_train_valid(input_path, output_dir, train_size=train_stories, valid_size=valid_stories)