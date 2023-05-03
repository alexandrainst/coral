"""Utility functions to be used in other ASR scripts."""

import json
from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig


def dump_vocabulary(cfg: DictConfig, dataset: Dataset) -> None:
    """Extracts the vocabulary from the dataset and dumps it to a file.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
        dataset (Dataset):
            The dataset from which to extract the vocabulary.
    """
    # Get all the text in the transcriptions. Note here that we have replaced the word
    # boundaries by pipes ("|"), so we are splitting on those.
    all_text = "|".join(dataset[cfg.dataset.text_column])

    # Build vocabulary
    unique_characters = set(all_text)
    vocab = {char: idx for idx, char in enumerate(unique_characters)}
    for tok in ["<unk>", "<pad>", "<s>", "</s>"]:
        vocab[tok] = len(vocab)

    # Dump the vocabulary to a json file
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = model_dir / "vocab.json"
    with vocab_path.open("w") as f:
        json.dump(vocab, f)
