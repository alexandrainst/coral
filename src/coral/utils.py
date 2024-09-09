"""General utility functions."""

import contextlib
import logging
import multiprocessing as mp
import os
import warnings
from functools import partialmethod
from pathlib import Path

import datasets.utils.logging as ds_logging
import tqdm as tqdm_package
import transformers.utils.logging as hf_logging
from datasets import (
    Dataset,
    IterableDataset,
    NamedSplit,
    disable_progress_bar,
    enable_progress_bar,
)
from huggingface_hub import CommitInfo, upload_folder
from tqdm.auto import tqdm
from transformers import Trainer


def block_terminal_output() -> None:
    """Blocks undesired terminal output.

    This blocks the following output:
        - User warnings
        - Logs from the `datasets` package
        - Progress bars from the `datasets` package
    """
    # Ignore user warnings throughout the codebase
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Disable logging from Hugging Face libraries
    ds_logging.set_verbosity_error()
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
    logging.getLogger("generation_whisper").setLevel(logging.ERROR)


class transformers_output_ignored:
    """Context manager to block terminal output."""

    def __enter__(self) -> None:
        """Enter the context manager."""
        hf_logging.set_verbosity_error()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        hf_logging.set_verbosity_info()


@contextlib.contextmanager
def monkeypatched(obj, name, patch):
    """Temporarily monkeypatch.

    Args:
        obj:
            The object to monkeypatch.
        name:
            The name of the attribute to monkeypatch.
        patch:
            The patch to apply.
    """
    old_attr = getattr(obj, name)
    setattr(obj, name, patch(old_attr))
    try:
        yield
    finally:
        setattr(obj, name, old_attr)


@contextlib.contextmanager
def disable_tqdm():
    """Context manager to disable tqdm."""

    def _patch(old_init):
        return partialmethod(old_init, disable=True)

    with monkeypatched(tqdm_package.std.tqdm, "__init__", _patch):
        yield


def convert_iterable_dataset_to_dataset(
    iterable_dataset: IterableDataset,
    split_name: str = "train",
    dataset_id: str | None = None,
) -> Dataset:
    """Convert an IterableDataset to a Dataset.

    Args:
        iterable_dataset:
            The IterableDataset to convert.
        split_name (optional):
            The name of the split. Defaults to "train".
        dataset_id (optional):
            The ID of the dataset, which is used to store and re-load the dataset. If
            None then the dataset is not stored. Defaults to None.

    Returns:
        The converted Dataset.
    """
    if dataset_id is not None:
        dataset_dir = Path.home() / ".cache" / "huggingface" / "datasets" / dataset_id
        if dataset_dir.exists():
            return Dataset.load_from_disk(str(dataset_dir))

    splits_info = iterable_dataset.info.splits
    num_examples = None if splits_info is None else splits_info[split_name].num_examples

    def gen_from_iterable_dataset():
        yield from tqdm(
            iterable=iterable_dataset,
            total=num_examples,
            desc="Converting iterable dataset to regular dataset",
        )

    with no_datasets_progress_bars():
        dataset = Dataset.from_generator(
            generator=gen_from_iterable_dataset,
            features=iterable_dataset.features,
            split=NamedSplit(name=split_name),
            num_proc=mp.cpu_count(),
        )
    assert isinstance(dataset, Dataset)

    if dataset_id is not None:
        dataset_dir.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(str(dataset_dir))

    return dataset


class no_datasets_progress_bars:
    """Context manager that disables the `datasets` progress bars."""

    def __enter__(self):
        """Disable the progress bar."""
        disable_progress_bar()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Re-enable the progress bar."""
        enable_progress_bar()


def interpret_dataset_name(dataset_name: str) -> tuple[str, str | None, str | None]:
    """Interpret the dataset name.

    This extracts the dataset ID, dataset subset and dataset revision from the dataset
    name.

    Args:
        dataset_name:
            The name of the dataset.

    Returns:
        A triple (dataset_id, dataset_subset, dataset_revision) where:
            dataset_id:
                The ID of the dataset.
            dataset_subset:
                The subset of the dataset, which can be None if the default subset
                should be used.
            dataset_revision:
                The revision of the dataset, which can be None if the newest revision
                should be used.
    """
    if ":" in dataset_name and "::" not in dataset_name:
        dataset_name = dataset_name.replace(":", "::")

    assert (
        dataset_name.count("@") <= 1
    ), "You cannot include more than one '@' in the dataset name"
    assert (
        dataset_name.count("::") <= 1
    ), "You cannot include more than one ':' in the dataset name"

    dataset_id = dataset_name
    dataset_subset = None
    dataset_revision = None

    if "@" in dataset_name:
        dataset_id_and_dataset_subset, dataset_revision_and_dataset_subset = (
            dataset_name.split("@")
        )
        if "::" in dataset_id_and_dataset_subset:
            dataset_id, dataset_subset = dataset_id_and_dataset_subset.split("::")
        else:
            dataset_id = dataset_id_and_dataset_subset
            dataset_subset = None
        if "::" in dataset_revision_and_dataset_subset:
            dataset_id, dataset_subset = dataset_revision_and_dataset_subset.split("::")
        else:
            dataset_revision = dataset_revision_and_dataset_subset

    if "::" in dataset_name:
        dataset_id, dataset_subset = dataset_name.split("::")
        if "@" in dataset_subset:
            dataset_subset, dataset_revision = dataset_subset.split("@")
        else:
            dataset_revision = None

    return dataset_id, dataset_subset, dataset_revision


def push_model_to_hub(
    trainer: Trainer,
    model_name: str,
    finetuned_from: str,
    language: str = "da",
    license: str = "openrail",
    tasks: list[str] = ["automatic-speech-recognition"],
    branch_name: str | None = None,
    create_pr: bool = False,
    commit_message: str = "Finished finetuning 🎉",
) -> CommitInfo | None:
    """Upload model and tokenizer to the Hugging Face Hub.

    This uses the model stored as `trainer.model` and the tokenizer stored as
    `trainer.tokenizer`, and uploads them to the model ID stored in
    `trainer.args.hub_model_id`.

    Args:
        trainer:
            The Trainer object containing the model and tokenizer to upload.
        model_name:
            The name of the model.
        finetuned_from:
            The ID of the model that was finetuned.
        language (optional):
            The language of the model. Defaults to "da" (Danish).
        license (optional):
            The license of the model. Defaults to "openrail".
        tasks (optional):
            The tasks the model is fine-tuned for. Defaults to
            ["automatic-speech-recognition"].
        branch_name (optional):
            The name of the branch to push to. Defaults to None, which means the
            default branch is used.
        create_pr (optional):
            Whether to create a pull request. Defaults to False.
        commit_message (optional):
            Message to commit while pushing. Defaults to "Finished finetuning 🎉".

    Returns:
        The commit information, or None if the process is not the main process.
    """
    token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

    # In case the user calls this method with trainer.args.push_to_hub = False
    if trainer.hub_model_id is None:
        trainer.init_hf_repo(token=token)

    # Only push from one node
    if not trainer.is_world_process_zero():
        return None

    trainer.create_model_card(
        model_name=model_name,
        language=language,
        license=license,
        tasks=tasks,
        finetuned_from=finetuned_from,
    )

    # Wait for the current upload to be finished.
    trainer._finish_current_push()
    return upload_folder(
        repo_id=trainer.hub_model_id or "",
        revision=branch_name,
        create_pr=create_pr,
        folder_path=trainer.args.output_dir,
        commit_message=commit_message,
        token=token or True,
        ignore_patterns=["_*", "checkpoint-*"],
    )
