import pyarrow.dataset as ds
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
import numpy as np
import torch


class FineWebDataset(IterableDataset):
    def __init__(self, batch_size=200):
        self.__ROOT_DIR = Path(__file__).resolve().parents[2]
        self.__PRETRAINING_DATASET_PATH = self.__ROOT_DIR / "datasets" / "pretraining"
        self.__dataset = ds.dataset(self.__PRETRAINING_DATASET_PATH, format="parquet")
        self.__batch_size = batch_size
    
    def __iter__(self):
        scanner = self.__dataset.scanner(columns=["text"], batch_size=self.__batch_size)
        for batch in scanner.to_batches():
            texts = batch.column("text").to_pylist()
            yield texts

    @staticmethod
    def get_dataloader(batch_size=200):
        """
        Returns a dataloader that returns python lists of size batch_size that contain text
        """
        return DataLoader(FineWebDataset(batch_size=batch_size), batch_size=None)
