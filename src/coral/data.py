"""Functions related to the data loading and processing."""

import io
import logging
import os
import re
import shutil
from collections.abc import Callable, Iterable, Sized
from functools import partial
from pathlib import Path
from typing import Any, Dict, cast
from unicodedata import normalize
from zipfile import ZipFile

import httpx
import torch
import torch_audiomentations as ta
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    NamedSplit,
    interleave_datasets,
    load_dataset,
)
from omegaconf import DictConfig
from tqdm.auto import tqdm

from .types import Data
from .utils import (
    NUMERAL_REGEX,
    convert_iterable_dataset_to_dataset,
    convert_numeral_to_words,
    interpret_dataset_name,
    no_datasets_progress_bars,
)

logger = logging.getLogger(__package__)


# Dictionary that contains characters to be converted (from the key to the value). Some
# values contain spaces to ensure that they're separated from other characters, and
# superfluous spaces are removed later. Note also that these are converted in the order
# they appear in the dictionary.
DEFAULT_CONVERSION_DICT = {
    "aa": "å",
    "ğ": "g",
    "ñ": "n",
    "ń": "n",
    "è": "e",
    "kg": " kilo ",
    "μg": " mikrogram ",
    "hhv": "henholdsvis",
    "fx": "for eksempel",
    "f.eks.": "for eksempel",
    "-": " minus ",
    "+": " plus ",
    "μ": " mikro ",
    "§": " paragraf ",
    "%": " procent ",
    "‰": " promille ",
    "ú": "u",
    "ş": "s",
    "ê": "e",
    "ã": "a",
    "ë": "e",
    "ć": "c",
    "ä": "æ",
    "í": "i",
    "š": "s",
    "î": "i",
    "ě": "e",
    "ð": "d",
    "á": "a",
    "ó": "o",
    "þ": "th",
    "ı": "i",
    "ö": "ø",
    "ç": "c",
    "ș": "s",
    "\u0301": " ",  # Empty whitespace symbol
    "\u200b": " ",  # Empty whitespace symbol
}


FILLER_WORDS_PATTERN = re.compile(
    pattern=r"\b(eh+m*|øh+m*|h+m+|m+h+)\b", flags=re.IGNORECASE
)


def load_data_for_finetuning(
    config: DictConfig, processor: Callable | None = None
) -> IterableDatasetDict:
    """Load an audio dataset for finetuning.

    Args:
        config:
            The Hydra configuration object.
        processor (optional):
            The processor to use for processing the audio and transcriptions. If `None`,
            then the processor is not used. Defaults to `None`.

    Returns:
        The audio dataset.

    Raises:
        ValueError:
            If the dataset is not supported.
    """
    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    all_datasets: list[IterableDataset] | list[Dataset] = list()
    len_datasets: list[int | None] = list()
    for dataset_name, dataset_config in config.datasets.items():
        if is_main_process:
            logger.info(f"Loading dataset {dataset_name!r}")

        # Load from disk if the dataset ID is a path and it is stored as an arrow
        # dataset
        if Path(dataset_config.id).exists():
            train_path = Path(dataset_config.id) / dataset_config.train_name
            data_files = list(map(str, train_path.glob("data-*.arrow")))
            if len(data_files) == 0:
                try:
                    ds = load_dataset(
                        path=dataset_config.id,
                        name=dataset_config.subset,
                        split=dataset_config.train_name,
                        streaming=config.streaming,
                        cache_dir=config.cache_dir,
                    )

                # In case a single split has been stored to disk, we load it directly
                except ValueError as e:
                    if "load_from_disk" not in str(e):
                        raise e
                    ds = Dataset.load_from_disk(dataset_path=dataset_config.id)
            else:
                try:
                    ds = load_dataset(
                        "arrow",
                        data_files=data_files,
                        split=dataset_config.train_name,
                        streaming=config.streaming,
                        cache_dir=config.cache_dir,
                    )
                except ValueError:
                    ds = load_dataset(
                        "arrow",
                        data_files=data_files,
                        split="train",
                        streaming=config.streaming,
                        cache_dir=config.cache_dir,
                    )

            if isinstance(ds, (IterableDataset, IterableDatasetDict)):
                length = None

                # info.splits is typed as "object", cast it to dict
                splits: Dict[str, object] = cast(
                    Dict[str, object], getattr(ds.info, "splits", {})
                )

                split_info = splits.get(dataset_config.train_name)
                if split_info is not None:
                    # num_examples might still be missing
                    length = getattr(split_info, "num_examples", None)

                len_datasets.append(length)

            else:
                # Non-streaming dataset: len() is safe
                len_datasets.append(len(ds))

        # Load dataset from the Hugging Face Hub. The HUGGINGFACE_HUB_TOKEN is only
        # used during CI - normally it is expected that the user is logged in to the
        # Hugging Face Hub using the `huggingface-cli login` command.
        else:
            with no_datasets_progress_bars():
                ds = load_dataset(
                    path=dataset_config.id,
                    name=dataset_config.subset,
                    split=dataset_config.train_name,
                    token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
                    streaming=config.streaming,
                    cache_dir=config.cache_dir,
                    trust_remote_code=True,
                )

        assert isinstance(ds, Dataset | IterableDataset), (
            f"Unsupported dataset type: {type(ds)}"
        )

        if dataset_config.text_column != "text":
            ds = ds.rename_column(dataset_config.text_column, "text")
        if dataset_config.audio_column != "audio":
            ds = ds.rename_column(dataset_config.audio_column, "audio")

        if dataset_config.filter_dataset:
            ds = filter_dataset(
                dataset=ds,
                audio_column="audio",
                text_column="text",
                min_seconds_per_example=config.min_seconds_per_example,
                max_seconds_per_example=config.max_seconds_per_example,
                is_main_process=is_main_process,
                num_proc=config.dataset_num_workers,
            )

        ds = ds.remove_columns(
            column_names=[
                column
                for column in ds.column_names or list()
                if column not in ["audio", "text"]
            ]
        ).shuffle(seed=config.seed)

        ds = ds.cast_column(
            column="audio", feature=Audio(sampling_rate=config.model.sampling_rate)
        )

        all_datasets.append(ds)  #  type: ignore[bad-argument-type]

    assert len(all_datasets) > 0, "No datasets were loaded"

    if len(all_datasets) > 1:
        if is_main_process:
            logger.info("Interleaving datasets...")
            if config.dataset_probabilities is None and len(all_datasets) > 1:
                logger.warning(
                    "No dataset probabilities were specified for the training split. "
                    "This means that each dataset will be sampled according to their "
                    "relative sizes, which might not be what you want."
                )

        probabilities = config.dataset_probabilities
        if probabilities is None:
            probabilities = [n / sum(len_datasets) for n in len_datasets]
            probabilities[-1] = 1 - sum(probabilities[:-1])

        elif sum(probabilities) != 1:
            raise ValueError(
                f"Dataset probabilities must sum to 1, but sum to {sum(probabilities)}"
            )

        assert len(all_datasets) == len(probabilities), (
            f"There are {len(all_datasets):,} datasets ({all_datasets}), but "
            f"{len(probabilities):,} probabilities ({probabilities}), but these "
            "should be equal!"
        )

        train = interleave_datasets(
            datasets=all_datasets,  #  type: ignore[bad-argument-type]
            probabilities=probabilities,
            seed=config.seed,
            split=NamedSplit("train"),
            stopping_strategy="all_exhausted",
        )
    else:
        train = all_datasets[0]

    train = process_dataset(
        dataset=train,
        lower_case=config.model.lower_case,
        characters_to_keep=config.model.characters_to_keep,
        text_column="text",
        audio_column="audio",
        convert_numerals=False,
        remove_input_dataset_columns=True,
        normalise_audio=True,
        augment_audio=True,
        processor=processor,
        num_proc=config.dataset_num_workers,
    )

    data_dict = dict(train=train)
    dataset = IterableDatasetDict(data_dict)

    if is_main_process:
        logger.info("Loading CoRal validation dataset...")

    def load_validation_dataset(dataset_config: DictConfig) -> Dataset:
        """Load a validation dataset.

        Args:
            dataset_config:
                The config for the dataset to load.

        Returns:
            The loaded dataset.
        """
        with no_datasets_progress_bars():
            val = load_dataset(
                path=dataset_config.id,
                name=dataset_config.subset,
                split=dataset_config.val_name,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
                streaming=True,
                cache_dir=config.cache_dir,
                trust_remote_code=True,
            )
        assert isinstance(val, IterableDataset)
        val = convert_iterable_dataset_to_dataset(
            iterable_dataset=val,
            split_name="val",
            dataset_id=dataset_config.id.replace("/", "--") + "-validation",
            cache_dir=config.cache_dir,
        )
        if dataset_config.text_column != "text":
            val = val.rename_column(dataset_config.text_column, "text")
        if dataset_config.audio_column != "audio":
            val = val.rename_column(dataset_config.audio_column, "audio")
        return val.cast_column(
            column="audio", feature=Audio(sampling_rate=config.model.sampling_rate)
        ).select_columns(column_names=["text", "audio"])

    vals = [
        load_validation_dataset(dataset_config=dataset_config)
        for dataset_config in config.evaluation_datasets
    ]
    vals = [
        filter_dataset(
            dataset=val,
            audio_column="audio",
            text_column="text",
            min_seconds_per_example=config.min_seconds_per_example,
            max_seconds_per_example=config.max_seconds_per_example,
            is_main_process=is_main_process,
            num_proc=config.dataset_num_workers,
        )
        for val in vals
    ]
    vals = [
        process_dataset(
            dataset=val,
            lower_case=config.evaluation_lower_case,
            characters_to_keep=config.evaluation_characters_to_keep,
            text_column="text",
            audio_column="audio",
            convert_numerals=False,
            remove_input_dataset_columns=True,
            normalise_audio=True,
            augment_audio=False,
            processor=processor,
            num_proc=config.dataset_num_workers,
        )
        for val in vals
    ]
    for dataset_config, split in zip(config.evaluation_datasets, vals):
        split_name = f"val_{dataset_config.id.split('/')[-1].lower().replace('-', '_')}"
        if dataset_config.subset is not None:
            split_name += f"_{dataset_config.subset.lower().replace('-', '_')}"
        dataset[split_name] = split

    return dataset


def load_dataset_for_evaluation(config: DictConfig) -> Dataset:
    """Load the evaluation dataset.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        A DatasetDict containing the validation and test datasets.
    """
    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    dataset_id, dataset_subset, dataset_revision = interpret_dataset_name(
        dataset_name=config.dataset
    )

    if is_main_process:
        logger.info(
            f"Loading the {config.eval_split_name!r} split of the {dataset_id} "
            "dataset..."
        )

    eval_dataset_path = None
    if config.cache_dir:
        eval_dataset_path = (
            Path(config.cache_dir) / "test-sets" / dataset_id.replace("/", "--")
        )
        if eval_dataset_path.exists():
            return Dataset.load_from_disk(dataset_path=eval_dataset_path)

    dataset = load_dataset(
        path=dataset_id,
        name=dataset_subset,
        split=config.eval_split_name,
        revision=dataset_revision,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
        cache_dir=config.cache_dir,
        streaming=True,
        trust_remote_code=True,
    )
    assert isinstance(dataset, IterableDataset)
    dataset = convert_iterable_dataset_to_dataset(
        iterable_dataset=dataset,
        split_name=config.eval_split_name,
        cache_dir=config.cache_dir,
    )
    assert isinstance(dataset, Dataset)
    dataset = filter_dataset(
        dataset=dataset,
        audio_column=config.audio_column,
        text_column=config.text_column,
        min_seconds_per_example=config.min_seconds_per_example,
        max_seconds_per_example=config.max_seconds_per_example,
        is_main_process=is_main_process,
    )
    dataset = dataset.cast_column(
        column=config.audio_column, feature=Audio(sampling_rate=config.sampling_rate)
    )
    dataset = process_dataset(
        dataset=dataset,
        lower_case=config.lower_case,
        characters_to_keep=config.characters_to_keep,
        remove_input_dataset_columns=False,
        text_column=config.text_column,
        audio_column=config.audio_column,
        normalise_audio=True,
        augment_audio=False,
        convert_numerals=True,
    )

    if eval_dataset_path is not None:
        dataset.save_to_disk(dataset_path=eval_dataset_path)

    return dataset


def filter_dataset(
    dataset: Data,
    audio_column: str,
    text_column: str,
    min_seconds_per_example: int | float,
    max_seconds_per_example: int,
    is_main_process: bool,
    num_proc: int | None = None,
) -> Data:
    """Filter the dataset.

    Note that this removes samples from the dataset.

    Args:
        dataset:
            The dataset to filter.
        audio_column:
            The name of the column containing the audio.
        text_column:
            The name of the column containing the transcriptions.
        min_seconds_per_example:
            The minimum number of seconds that an example can have.
        max_seconds_per_example:
            The maximum number of seconds that an example can have.
        is_main_process:
            Whether the current process is the main process.
        num_proc (optional):
            The number of processes to use for filtering the dataset. If `None`, then
            no multiprocessing is used. Defaults to `None`.

    Returns:
        The filtered dataset.
    """
    num_samples_before = len(dataset) if isinstance(dataset, Sized) else 0

    filter_fn = partial(
        filter_example,
        text_column=text_column,
        audio_column=audio_column,
        min_seconds_per_example=min_seconds_per_example,
        max_seconds_per_example=max_seconds_per_example,
    )
    if isinstance(dataset, Dataset | DatasetDict):
        filtered = dataset.filter(
            function=filter_fn,
            num_proc=num_proc,
            desc="Filtering dataset",
            keep_in_memory=True,
        )
    else:
        filtered = dataset.filter(function=filter_fn)

    # Add info back in the filtered dataset, as it gets removed after calling `filter`
    if isinstance(dataset, Dataset | IterableDataset) and isinstance(
        filtered, Dataset | IterableDataset
    ):
        filtered.info.features = dataset.info.features
    else:
        assert isinstance(dataset, DatasetDict | IterableDatasetDict) and isinstance(
            filtered, DatasetDict | IterableDatasetDict
        )
        for split_name in dataset.keys():
            dataset[split_name].info.features = filtered[split_name].info.features

    if isinstance(dataset, Sized) and isinstance(filtered, Sized) and is_main_process:
        num_samples_removed = num_samples_before - len(filtered)
        logger.info(f"Removed {num_samples_removed:,} samples from the dataset")

    return filtered  #  type: ignore[bad-return]


def filter_example(
    sample: dict[str, Any],
    audio_column: str,
    text_column: str,
    min_seconds_per_example: int | float,
    max_seconds_per_example: int,
) -> bool:
    """Filter samples based on the validation status.

    Args:
        sample:
            The sample to filter.
        audio_column:
            The name of the column containing the audio.
        text_column:
            The name of the column containing the transcriptions.
        min_seconds_per_example:
            The minimum number of seconds that an example can have.
        max_seconds_per_example:
            The maximum number of seconds that an example can

    Returns:
        Whether the sample should be kept.
    """
    # Filtering based on audio
    audio = sample[audio_column]
    if audio["array"].shape[0] <= audio["sampling_rate"] * min_seconds_per_example:
        return False
    if audio["array"].shape[0] >= audio["sampling_rate"] * max_seconds_per_example:
        return False

    # Filtering based on text
    if len(sample[text_column].strip()) == 0:
        return False

    # Filtering based on validation
    if "validated" in sample and sample["validated"] == "rejected":
        return False

    return True


def process_dataset(
    dataset: Data,
    lower_case: bool,
    characters_to_keep: Iterable[str] | None,
    remove_input_dataset_columns: bool,
    text_column: str,
    audio_column: str | None,
    convert_numerals: bool,
    normalise_audio: bool,
    augment_audio: bool,
    num_proc: int | None = None,
    processor: Callable | None = None,
) -> Data:
    """Process the dataset.

    Note that this does not remove any samples from the dataset.

    Args:
        dataset:
            The dataset to be cleaned.
        lower_case:
            Whether to make the text lower case.
        characters_to_keep:
            All the characters that should be kept in the transcriptions. Can be None if
            all characters should be kept.
        text_column:
            The name of the column containing the text.
        remove_input_dataset_columns:
            Whether to remove all input dataset columns from the output dataset.
        audio_column:
            The name of the column containing the audio. Can be `None` if the dataset
            does not have an audio column.
        convert_numerals:
            Whether to convert numerals to words.
        normalise_audio:
            Whether to normalise the audio.
        augment_audio:
            Whether to augment the audio.
        num_proc (optional):
            The number of processes to use for processing the dataset. If `None`, then
            no multiprocessing is used. Defaults to `None`.
        processor (optional):
            The processor to use for processing the audio and transcriptions. If `None`,
            then the processor is not used. Defaults to `None`.

    Returns:
        The cleaned dataset.
    """
    if isinstance(dataset, Dataset) or isinstance(dataset, IterableDataset):
        column_names = dataset.column_names
    elif isinstance(dataset, DatasetDict) or isinstance(dataset, IterableDatasetDict):
        column_names = dataset["train"].column_names
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    map_fn = partial(
        process_example,
        characters_to_keep=characters_to_keep,
        conversion_dict=DEFAULT_CONVERSION_DICT,
        text_column=text_column,
        audio_column=audio_column,
        lower_case=lower_case,
        convert_numerals=convert_numerals,
        processor=processor,
        normalise_audio=normalise_audio,
        augment_audio=augment_audio,
    )
    if isinstance(dataset, Dataset | DatasetDict):
        mapped = dataset.map(
            function=map_fn,
            num_proc=num_proc,
            desc="Processing dataset",
            remove_columns=column_names if remove_input_dataset_columns else None,
        )
    else:
        mapped = dataset.map(function=map_fn, remove_columns=column_names)

    return mapped  #  type: ignore[bad-return]


def process_example(
    example: dict,
    characters_to_keep: Iterable[str] | None,
    conversion_dict: dict[str, str],
    text_column: str,
    audio_column: str | None,
    lower_case: bool,
    convert_numerals: bool,
    processor: Callable | None,
    normalise_audio: bool,
    augment_audio: bool,
) -> dict:
    """Helper function which cleans a single example.

    Args:
        example:
            The example to be cleaned.
        characters_to_keep:
            All the characters that should be kept in the transcriptions. Can be None if
            all characters should be kept.
        conversion_dict:
            A dictionary of characters to be converted.
        text_column:
            The name of the column containing the text.
        audio_column:
            The name of the column containing the audio. Can be `None` if the dataset
            does not have an audio column.
        lower_case:
            Whether to make the text lower case.
        convert_numerals:
            Whether to convert numerals to words.
        processor:
            The processor to use for processing the audio and transcriptions. If `None`,
            then the processor is not used. Requires `audio_column` to be specified.
        normalise_audio:
            Whether to normalise the audio.
        augment_audio:
            Whether to augment the audio.

    Returns:
        The cleaned example.
    """
    doc = example[text_column]

    if convert_numerals and re.search(pattern=NUMERAL_REGEX, string=doc):
        doc = "".join(
            convert_numeral_to_words(numeral=maybe_numeral)
            for maybe_numeral in re.split(pattern=NUMERAL_REGEX, string=doc)
            if maybe_numeral is not None
        )

    if lower_case:
        doc = doc.lower()

    # Remove filler words such as "ehh"
    doc = FILLER_WORDS_PATTERN.sub(repl="", string=doc)

    # Normalise the transcription, which uniformises the characters. For instance, the
    # "long dash" (－) is converted to the normal dash (-).
    doc = normalize("NFKC", doc)

    # Convert known symbols
    for key, value in conversion_dict.items():
        doc = doc.replace(key, value)

    # Remove all non-standard characters
    if characters_to_keep is not None:
        characters_to_keep = "".join(char for char in characters_to_keep)
        non_standard_characters_regex = re.compile(
            f"[^{re.escape(characters_to_keep + ' |')}]", flags=re.IGNORECASE
        )
        doc = re.sub(non_standard_characters_regex, " ", doc.strip())

    # Replace superfluous spaces
    doc = re.sub(r" +", " ", doc)

    # Strip each newline
    doc = "\n".join([line.strip() for line in doc.split("\n")]).strip("\n")

    # Re-assign the cleaned transcription
    example[text_column] = doc

    # If we do not have any audio, then we return the example, as the remainder of the
    # function concerns audio processing
    if audio_column is None:
        return example

    # Extract audio from example
    audio = example[audio_column]
    audio_array = audio["array"]
    sampling_rate = audio["sampling_rate"]

    # Normalise and augment audio
    download_background_noises()
    normalise = ta.PeakNormalization(p=1.0) if normalise_audio else ta.Identity()
    augment = (
        ta.Compose(
            [
                ta.PeakNormalization(p=1.0),
                ta.Gain(p=1.0),
                ta.AddBackgroundNoise(
                    background_paths=Path("background-noises"), p=0.7
                ),
                ta.AddColoredNoise(p=0.2),
                ta.OneOf(
                    [
                        ta.BandPassFilter(p=1.0),
                        ta.BandStopFilter(p=1.0),
                        ta.HighPassFilter(p=1.0),
                        ta.LowPassFilter(p=1.0),
                    ],
                    p=0.2,
                ),
            ],
            p=1.0,
        )
        if augment_audio
        else ta.Identity()
    )
    normalise_and_augment = ta.Compose([normalise, augment], p=1.0)
    audio_array = normalise_and_augment(
        torch.tensor(audio_array).unsqueeze(0).unsqueeze(0), sample_rate=sampling_rate
    )[0, 0]

    # If we don't have a processor then we just re-assign the normalised audio and
    # return the processed example
    if processor is None:
        example[audio_column]["array"] = audio_array
        return example

    # Process the audio
    processed = processor(audio_array, sampling_rate=sampling_rate)
    audio_feature_name = (
        "input_values" if "input_values" in processed else "input_features"
    )
    audio_array = processed[audio_feature_name][0]
    example[audio_feature_name] = audio_array
    example["num_seconds"] = len(example[audio_feature_name]) / sampling_rate

    # Process the labels
    example["labels"] = processor(text=example[text_column], truncation=True).input_ids
    example["input_length"] = len(example["labels"])

    return example


def download_background_noises() -> None:
    """Download background noises for audio augmentation.

    This function downloads the background noises to the `background-noises` directory,
    and will do nothing if the directory already exists.
    """
    background_noises_path = Path("background-noises")
    if background_noises_path.exists():
        return

    logger.info("Downloading background noises from the ESC-50 dataset...")

    # Download the ESC-50 dataset zip file as a stream
    zip_url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    chunks = []
    with httpx.stream(method="GET", url=zip_url, follow_redirects=True) as response:
        for chunk in tqdm(
            response.iter_bytes(),
            desc="Downloading ESC-50 dataset",
            unit="B",
            unit_scale=True,
            total=int(response.headers.get("Content-Length", 0)),
        ):
            chunks.append(chunk)
    content = b"".join(chunks)

    # Unzip only the audio files from the ESC-50 dataset
    with ZipFile(file=io.BytesIO(content)) as zip_file:
        audio_files = [
            file_info
            for file_info in zip_file.infolist()
            if file_info.filename.startswith("ESC-50-master/audio/")
        ]
        zip_file.extractall(members=audio_files, path=background_noises_path)

    # Move audio files to the root of the background-noises directory
    extracted_audio_path = background_noises_path / "ESC-50-master" / "audio"
    for audio_file in extracted_audio_path.iterdir():
        audio_file.rename(background_noises_path / audio_file.name)

    # Remove the extracted directories
    shutil.rmtree(background_noises_path / "ESC-50-master")

    logger.info("Background noises downloaded successfully.")
