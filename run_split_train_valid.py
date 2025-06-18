from pathlib import Path
import random


def split_train_valid(input_path: Path, output_dir: Path, valid_size: int = 200000, seed: int = 2024):
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read stories in chunks and count total
    stories = []
    delimiter = "<|endoftext|>"
    current_story = ""
    total_stories = 0

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        while chunk := f.read(256 * 256):  # 1MB chunks
            current_story += chunk
            parts = current_story.split(delimiter)
            current_story = parts[-1]
            for part in parts[:-1]:
                if part.strip():
                    stories.append(part.strip())
                    total_stories += 1
            if total_stories % 100000 == 0:
                print(f"Processed {total_stories} stories")

    if current_story.strip():
        stories.append(current_story.strip())
        total_stories += 1

    print(f"Total stories: {total_stories}")

    # Sample validation indices
    valid_indices = random.sample(range(total_stories), min(valid_size, total_stories))
    valid_indices_set = set(valid_indices)

    # Write train and valid files
    with open(output_dir / "train.txt", "w", encoding="utf-8") as train_f, \
            open(output_dir / "valid.txt", "w", encoding="utf-8") as valid_f:
        for i, story in enumerate(stories):
            if i in valid_indices_set:
                valid_f.write(story + delimiter)
            else:
                train_f.write(story + delimiter)

    print(f"Saved {len(valid_indices)} stories to valid.txt, {total_stories - len(valid_indices)} to train.txt")


if __name__ == "__main__":
    input_path = Path("data/TinyStoriesV2-GPT4/train.txt")
    output_dir = Path("data/TinyStoriesV2-GPT4/split")
    split_train_valid(input_path, output_dir, valid_size=300000)
