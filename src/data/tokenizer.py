"""
BPE tokenizer training and inference for SVG text.
"""

import json
import logging
import os
from pathlib import Path
from typing import Union

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from src.utils.config import get_config


logger = logging.getLogger(__name__)

config = get_config("data/data")

# Default vocab size for BPE tokenizer
VOCAB_SIZE = config["tokenizer"]["vocab_size"]

# Default special tokens
UNK_TOKEN = config["tokenizer"]["special_tokens"]["unk"]
BOS_TOKEN = config["tokenizer"]["special_tokens"]["bos"]
EOS_TOKEN = config["tokenizer"]["special_tokens"]["eos"]
SPECIAL_TOKENS = [UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# Directory to save/load tokenizer files
TOK_DIR = Path(config["data_paths"]["tokenizer_dir"])

class SVGTokenizer:
    """BPE tokenizer for SVG strings using HuggingFace tokenizers library."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self._tokenizer = None  # populated by train() or from_pretrained()

    def train(self, texts: list[str]) -> None:
        """
        Train a BPE tokenizer on the provided texts.
        """
        tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        # Pre-tokenizers split on whitespace and split digits from letters
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Digits(individual_digits=False),
        ])

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=1,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )

        logger.info("Training BPE tokenizer...")
        tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
        self._tokenizer = tokenizer
        logger.info(f"Tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")

    def save(self, directory: Union[str, os.PathLike] = TOK_DIR) -> None:
        """Save tokenizer files to the specified directory."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer has not been trained yet.")
        
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path / "tokenizer.json"))

        # Save a config file with vocab size and special tokens
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "bos_token": BOS_TOKEN,
            "eos_token": EOS_TOKEN,
            "unk_token": UNK_TOKEN,
        }
        with open(path / "tokenizer_config.json", "w") as fh:
            json.dump(config, fh, indent=2)

        logger.info(f"Tokenizer saved to {path}")

    @classmethod
    def from_pretrained(cls, directory: Union[str, os.PathLike] = TOK_DIR) -> "SVGTokenizer":
        """Load a previously saved tokenizer from a specified directory."""
        path = Path(directory)
        tokenizer_path = path / "tokenizer.json"
        config_path    = path / "tokenizer_config.json"

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"No trained tokenizer found at '{tokenizer_path}'. "
                "Run the data preparation pipeline first (prepare_data.py)."
            )
        if not config_path.exists():
            raise FileNotFoundError(
                f"No tokenizer config found at '{config_path}'. "
                "Run the data preparation pipeline first (prepare_data.py)."
            )

        with open(config_path) as fh:
            config = json.load(fh)
        vocab_size = config["vocab_size"]

        obj = cls(vocab_size=vocab_size)
        obj._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        logger.info(
            f"Loaded tokenizer from {path} (vocab_size={obj._tokenizer.get_vocab_size()})"
        )
        return obj

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode a single SVG string to a list of token ids."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer has not been trained/loaded yet.")

        enc = self._tokenizer.encode(text)
        ids: list[int] = enc.ids

        if add_special_tokens:
            bos_id = self._tokenizer.token_to_id(BOS_TOKEN)
            eos_id = self._tokenizer.token_to_id(EOS_TOKEN)
            ids = [bos_id] + ids + [eos_id]

        return ids

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """Encode a list of SVG strings in parallel."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer has not been trained/loaded yet.")

        encodings = self._tokenizer.encode_batch(texts)
        results = []
        bos_id = self._tokenizer.token_to_id(BOS_TOKEN)
        eos_id = self._tokenizer.token_to_id(EOS_TOKEN)

        for enc in encodings:
            ids = enc.ids
            if add_special_tokens:
                ids = [bos_id] + ids + [eos_id]
            results.append(ids)

        return results

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token ids back to a string."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer has not been trained/loaded yet.")
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def actual_vocab_size(self) -> int:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized.")
        return self._tokenizer.get_vocab_size()

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.token_to_id(BOS_TOKEN)

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.token_to_id(EOS_TOKEN)
