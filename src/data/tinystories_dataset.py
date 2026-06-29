from pathlib import Path
from torch.utils.data.dataset import Dataset
from src.tokenizer.tokenizer import Tokenizer
from typing import List, Tuple
import torch

class TinyStoriesDataset(Dataset):
    """
    Returns the next entry from the TinyStories datasets
    """
    def __init__(self, tokenizer: Tokenizer, block_size: int = 512, train: bool = True):
        self.ROOT_DIR = Path(__file__).resolve().parents[2]
        self.TINYSTORIES_DATASET_PATH = self.ROOT_DIR / "datasets" / "tinystories"
        if train:
            self.TINYSTORIES_DATASET_PATH = self.TINYSTORIES_DATASET_PATH / "TinyStories-train.txt"
        else:
            self.TINYSTORIES_DATASET_PATH = self.TINYSTORIES_DATASET_PATH / "TinyStories-valid.txt"

        self.block_size = block_size
        self.tokenizer = tokenizer
        self.eot = "<|endoftext|>"
        self.spans = self._build_spans()

    def _build_spans(self) -> List[Tuple[int, int]]:
        """
        Builds byte spans by scanning for EOT token in binary mode.
        """
        delim = self.eot.encode("utf-8")
        spans: List[Tuple[int, int]] = []

        with open(self.TINYSTORIES_DATASET_PATH, "rb") as f:
            data = f.read()
        
        self.length_longest = 0
        start = 0
        while True:
            idx = data.find(delim, start)
            if idx == -1:
                end = len(data)
                spans.append((start, end))
                break
            spans.append((start, idx + len(delim)))
            start = idx + len(delim)

        return spans

    def __len__(self):
        return len(self.spans)
    
    def _read_entry(self, idx: int) -> str:
        start, end = self.spans[idx]
        with open(self.TINYSTORIES_DATASET_PATH, "rb") as f:
            f.seek(start)
            raw = f.read(end - start)
        text = raw.decode("utf-8").strip()
        return text


    def __getitem__(self, idx: int):
        text = self._read_entry(idx)

        ids = self.tokenizer.tokenize_to_ids(text)

        pad_id = self.tokenizer.get_pad_token_id()

        ids = ids[:self.block_size]
        if len(ids) < self.block_size:
            ids = ids + [pad_id] * (self.block_size - len(ids))

        ids_tensor = torch.tensor(ids, dtype=torch.long)

        labels = ids_tensor.clone()
        labels[:-1] = ids_tensor[1:]
        labels[-1] = -100
        labels[ids_tensor == pad_id] = -100

        return ids_tensor, labels
