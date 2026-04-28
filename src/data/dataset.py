"""
Dataset for token-level language modeling on SVG text.
"""

import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import get_config


logger = logging.getLogger(__name__)

config = get_config("data/data")

# Context window size
BLOCK_SIZE = config["block_size"]


class SVGDataset(Dataset):
    """PyTorch dataset for language modeling on SVG text."""

    def __init__(
        self,
        token_ids: Union[list[int], torch.Tensor],
        block_size: int = BLOCK_SIZE,
    ):
        """
        Args: 
            token_ids: Flat list or 1D tensor of all token ids for the split
            block_size: Context window size
                Input is ids[i:i+block_size], target is ids[i+1:i+block_size+1]
        """
        if isinstance(token_ids, torch.Tensor):
            self.data = token_ids.long()
        else:
            self.data = torch.tensor(token_ids, dtype=torch.long)

        self.block_size = block_size
        
        logger.info(
            f"SVGDataset: {len(self.data)} tokens, block_size={block_size} -> {len(self)} chunks"
        )

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # A chunk represents one training example
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    @classmethod
    def from_svgs(
        cls,
        svg_strings: list[str],
        tokenizer,
        block_size: int = BLOCK_SIZE,
        add_special_tokens: bool = True,
    ) -> "SVGDataset":
        """Tokenize a list of SVG strings and build a Pytorch dataset."""
        logger.info(f"Tokenizing {len(svg_strings)} SVGs...")

        all_ids: list[int] = []
        max_token_length = config["cleaning"]["max_token_length"]
        seq_lengths: list[int] = []
        n_filtered = 0
        for ids in tokenizer.encode_batch(
            svg_strings, add_special_tokens=add_special_tokens
        ):
            if len(ids) > max_token_length:
                n_filtered += 1
                continue
            seq_lengths.append(len(ids))
            all_ids.extend(ids)

        if n_filtered:
            logger.info(f"Filtered {n_filtered}/{len(svg_strings)} SVGs exceeding {max_token_length} tokens.")
        logger.info(f"Total tokens: {len(all_ids)}")

        dataset = cls(all_ids, block_size=block_size)
        dataset.seq_lengths = seq_lengths
        dataset.n_filtered = n_filtered
        return dataset

    @classmethod
    def from_file(cls, path: Union[str, os.PathLike], block_size: int = BLOCK_SIZE) -> "SVGDataset":
        """Load a tokenized .bin file."""
        arr = np.memmap(path, dtype=np.int32, mode="r")
        data = torch.from_numpy(arr.astype(np.int64))
        return cls(data, block_size=block_size)

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Save the token tensor to a .bin file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arr = np.memmap(path, dtype=np.int32, mode="w+", shape=(len(self.data),))
        arr[:] = self.data.numpy().astype(np.int32)
        arr.flush()

        logger.info(f"Saved {len(self.data)} tokens to {path}")
