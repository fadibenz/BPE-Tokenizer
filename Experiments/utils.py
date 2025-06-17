import regex as re
import json
import re
import random
from typing import List
from pathlib import Path

def sample_stories(
        path: Path,
        number: int = 1000,
        delimiter: str = "<|endoftext|>",
) -> List[str]:
    if number <= 0:
        raise ValueError("Number of samples must be positive")


    path = Path(path) if isinstance(path, str) else path

    # Reservoir sampling for file-based corpus
    reservoir = []
    current_story = ""
    story_count = 0
    pattern = re.compile(re.escape(delimiter))

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            current_story += line
            parts = pattern.split(current_story)
            if len(parts) > 1:
                # Process complete stories
                for part in parts[:-1]:
                    story = part.strip()
                    if story:
                        story_count += 1
                        if len(reservoir) < number:
                            reservoir.append(story)
                        else:
                            if random.random() < number / story_count:
                                reservoir[random.randint(0, number - 1)] = story
                current_story = parts[-1]

    if current_story.strip():
        story_count += 1
        if len(reservoir) < number:
            reservoir.append(current_story.strip())
        else:
            if random.random() < number / story_count:
                reservoir[random.randint(0, number - 1)] = current_story.strip()

    return reservoir

def log_stats(stats, output_path):

    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
