from __future__ import annotations

from pathlib import Path

from src.data.tinystories_dataset import TinyStoriesDataset


def build_dataset(tokenizer, path: Path, block_size: int, train: bool) -> TinyStoriesDataset:
    expected_name = "TinyStories-train.txt" if train else "TinyStories-valid.txt"
    if path.name != expected_name:
        raise ValueError(f"expected {expected_name}, got {path.name}")
    return TinyStoriesDataset(tokenizer=tokenizer, block_size=block_size, train=train)
