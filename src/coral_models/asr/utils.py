"""Utility functions to be used in other ASR scripts."""

import json
from pathlib import Path

from datasets import Dataset, IterableDataset
from omegaconf import DictConfig


def dump_vocabulary(cfg: DictConfig, dataset: Dataset | IterableDataset) -> None:
    """Extracts the vocabulary from the dataset and dumps it to a file.

    It will dump the file to `${cfg.model_dir}/vocab.json`.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
        dataset (Dataset or IterableDataset):
            The dataset from which to extract the vocabulary.
    """
    # Build the set of all unique characters in the dataset
    unique_characters = {"|"}
    mapped_dataset = dataset.remove_columns("audio").map(
        lambda exs: unique_characters.update("".join(exs[cfg.data.text_column])),
        batched=True,
    )

    # If the dataset is iterable then the `map` method is lazy and we need to iterate
    # over it to actually execute the mapping
    if isinstance(mapped_dataset, IterableDataset):
        for _ in mapped_dataset:
            pass

    # Build vocabulary
    vocab = {char: idx for idx, char in enumerate(unique_characters)}
    for tok in ["<unk>", "<pad>", "<s>", "</s>"]:
        vocab[tok] = len(vocab)

    # Dump the vocabulary to a json file
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = model_dir / "vocab.json"
    with vocab_path.open("w") as f:
        json.dump(vocab, f)
