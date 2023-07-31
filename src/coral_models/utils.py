"""General utility functions."""

import json
import warnings
from pathlib import Path

import datasets.utils.logging as ds_logging
import transformers.utils.logging as hf_logging
from datasets.utils import disable_progress_bar
from omegaconf import DictConfig


def block_terminal_output() -> None:
    """Blocks undesired terminal output.

    This blocks the following output:
        - User warnings
        - Logs from the `datasets` package
        - Progress bars from the `datasets` package
    """

    # Ignore user warnings throughout the codebase
    warnings.filterwarnings("ignore", category=UserWarning)

    # Disable logging from the `datasets` library
    ds_logging.set_verbosity_error()

    # Disable the tokeniser progress bars from the `datasets` library
    disable_progress_bar()


class transformers_output_ignored:
    """Context manager to block terminal output."""

    def __enter__(self) -> None:
        hf_logging.set_verbosity_error()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        hf_logging.set_verbosity_info()


def dump_vocabulary(cfg: DictConfig) -> None:
    """Extracts the vocabulary from the dataset and dumps it to a file.

    It will dump the file to `${cfg.model_dir}/vocab.json`.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
    """
    # Build the set of all unique characters in the dataset
    unique_characters: set[str] = set(cfg.characters_to_keep)

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
