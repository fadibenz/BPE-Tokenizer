import numpy as np
import regex as re
import json

def sample_stories(path,
                   number:int = 1000):

    with open(path / "valid.txt", "r", encoding="utf-8") as f:
        file = f.read()

    stories = np.array(re.split(r"<\|endoftext\|>", file))
    indices = np.random.randint(0, len(stories), number)

    return stories[indices]

def log_stats(stats, output_path):

    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
