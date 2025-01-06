"""Script that builds and uploads the CoRal speech recognition dataset from the raw data.

Usage:
    python src/scripts/build_coral_asr.py [key=value] [key=value] ...
"""

import logging
import multiprocessing as mp
import shutil
import sqlite3
import tarfile
from functools import partial
from pathlib import Path
from time import sleep

import hydra
from datasets import Audio, Dataset, DatasetDict
from joblib import Parallel, delayed
from omegaconf import DictConfig
from requests import HTTPError
from tqdm.auto import tqdm

from coral.utils import no_datasets_progress_bars
from coral.validation import add_validations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_coral_asr")


@hydra.main(
    config_path="../../config", config_name="dataset_creation", version_base=None
)
def main(config: DictConfig) -> None:
    """Build and upload the CoRal speech recognition dataset.

    Args:
        config:
            The Hydra configuration object
    """
    metadata_database_path = Path(config.metadata_database_path)
    read_aloud_dir = Path(config.audio_dir) / "recordings"
    conversation_dir = Path(config.audio_dir) / "conversations"

    logger.info("Copying the raw files to the current working directory...")
    temp_read_aloud_dir = copy_audio_directory_to_cwd(audio_dir=read_aloud_dir)
    temp_conversation_dir = copy_audio_directory_to_cwd(audio_dir=conversation_dir)
    temp_metadata_database_path = Path.cwd() / metadata_database_path.name
    shutil.copy(src=metadata_database_path, dst=temp_metadata_database_path)

    # Copy the metadata database to the current working directory, since that massively
    # speeds up the SQL queries
    logger.info("Copying the metadata database to the current working directory...")
    temp_metadata_database_path = Path.cwd() / metadata_database_path.name
    shutil.copy(src=metadata_database_path, dst=temp_metadata_database_path)

    logger.info("Building the CoRal read-aloud speech recognition dataset...")
    read_aloud_dataset = build_read_aloud_dataset(
        metadata_database_path=temp_metadata_database_path,
        audio_dir=temp_read_aloud_dir,
    )

    logger.info("Building the CoRal conversation speech recognition dataset...")
    conversation_dataset = build_conversation_dataset(
        metadata_database_path=temp_metadata_database_path,
        audio_dir=temp_conversation_dir,
    )

    logger.info("Validating and filtering the datasets...")
    read_aloud_dataset = add_validations(
        dataset=read_aloud_dataset,
        text_column="text",
        audio_column="audio",
        model_id=config.validation.model_id,
        clean_text=config.validation.clean_text,
        lower_case=config.validation.lower_case,
        sampling_rate=config.validation.sampling_rate,
        characters_to_keep=config.validation.characters_to_keep,
        batch_size=config.validation.batch_size,
        max_cer=config.validation.max_cer,
    )
    conversation_dataset = add_validations(
        dataset=conversation_dataset,
        text_column="text",
        audio_column="audio",
        model_id=config.validation.model_id,
        clean_text=config.validation.clean_text,
        lower_case=config.validation.lower_case,
        sampling_rate=config.validation.sampling_rate,
        characters_to_keep=config.validation.characters_to_keep,
        batch_size=config.validation.batch_size,
        max_cer=config.validation.max_cer,
    )

    logger.info("Splitting the datasets into train, validation and test sets...")
    read_aloud_dataset = split_dataset(
        dataset=read_aloud_dataset,
        test_speakers=config.test_speakers,
        val_speakers=config.val_speakers,
    )
    conversation_dataset = split_dataset(
        dataset=conversation_dataset,
        test_speakers=config.test_speakers,
        val_speakers=config.val_speakers,
    )

    logger.info(
        f"Uploading the datasets to {config.hub_id!r} on the Hugging Face Hub..."
    )
    upload_dataset(
        read_aloud_dataset=read_aloud_dataset,
        conversation_dataset=conversation_dataset,
        hub_id=config.hub_id,
    )

    logger.info(
        f"All done! See the datasets at https://hf.co/datasets/{config.hub_id}."
    )


##########################################
##### Building the read-aloud subset #####
##########################################


def build_read_aloud_dataset(metadata_database_path: Path, audio_dir: Path) -> Dataset:
    """Build the CoRal read-aloud dataset.

    Args:
        metadata_database_path:
            Path to the SQLite database containing the metadata.
        audio_dir:
            Path to the directory containing the audio files.

    Returns:
        The CoRal read-aloud dataset.
    """
    # Get the number of samples in the SQLite database. We don't do any merges here to
    # save some time. That means that the count will be an upper bound rather than a
    # precise number of samples, but we deal with that when we actually fetch the data
    count_query = "SELECT COUNT(*) FROM Recordings;"
    with sqlite3.connect(database=metadata_database_path) as connection:
        cursor = connection.cursor()
        cursor.execute(count_query)
        num_metadata_samples = cursor.fetchone()[0]
    logger.info(f"There are {num_metadata_samples:,} samples in the SQLite database.")

    # Set up which features to fetch from the SQLite database. We exclude the ID
    # features since they need to be handled separately
    non_id_features = [
        "datetime_start",
        "datetime_end",
        "text",
        "location",
        "location_roomdim",
        "noise_level",
        "noise_type",
        "source_url",
        "age",
        "gender",
        "dialect",
        "language_native",
        "language_spoken",
        "country_birth",
        "zipcode_birth",
        "zip_school",
        "education",
        "occupation",
        "validated",
    ]
    non_id_features_str = ",\n".join(non_id_features)

    selection_query = f"""
        SELECT
            Recordings.id_recording,
            Sentences.id_sentence,
            Speakers.id_speaker,
            Recordings.id_validator,
            {non_id_features_str}
        FROM
            Recordings
            INNER JOIN Sentences ON Recordings.id_sentence = Sentences.id_sentence
            INNER JOIN Speakers ON Recordings.id_speaker = Speakers.id_speaker
    """

    # Open the database connection and fetch the data
    logger.info("Fetching the metadata from the SQLite database...")
    with sqlite3.connect(database=metadata_database_path) as connection:
        cursor = connection.cursor()
        cursor.execute(selection_query)
        rows = list(map(list, cursor.fetchall()))

    # Get a list of all the audio file paths. We need this since the audio files lie in
    # subdirectories of the main audio directory
    audio_subdirs = list(audio_dir.iterdir())
    with Parallel(n_jobs=mp.cpu_count(), backend="threading") as parallel:
        all_audio_path_lists = parallel(
            delayed(list_audio_files)(subdir)
            for subdir in tqdm(audio_subdirs, desc="Collecting audio file paths")
        )
    all_audio_paths = {
        path.stem: path
        for path_list in all_audio_path_lists
        for path in path_list or []
    }

    # Match the audio files to the metadata, to ensure that there is a 1-to-1
    # correspondence between them
    logger.info("Matching the audio files to the metadata...")
    recording_ids: list[str] = [row[0] for row in rows]
    matched_audio_paths = [
        all_audio_paths.get(recording_id) for recording_id in recording_ids
    ]
    rows = [
        row + [str(audio_path)]
        for row, audio_path in zip(rows, matched_audio_paths)
        if audio_path is not None
    ]

    # Build the dataset from the metadata and the audio files. This embeds all the audio
    # files into the dataset as parquet files
    dataset = Dataset.from_dict(
        mapping={
            "id_recording": [row[0] for row in rows],
            "id_sentence": [row[1] for row in rows],
            "id_speaker": [row[2] for row in rows],
            "id_validator": [row[3] for row in rows],
            **{
                feature: [row[i] for row in rows]
                for i, feature in enumerate(non_id_features, start=4)
            },
            "audio": [row[-1] for row in rows],
        }
    )
    dataset = dataset.cast_column("audio", Audio())
    return dataset


def list_audio_files(audio_dir: Path, max_attempts: int = 10) -> list[Path]:
    """List all the audio files in the given directory.

    Args:
        audio_dir:
            The directory containing the audio files.
        max_attempts (optional):
            The maximum number of attempts to list the audio files. Defaults to 10.

    Returns:
        A list of paths to the audio files.

    Raises:
        OSError:
            If the audio files cannot be listed.
    """
    for _ in range(max_attempts):
        try:
            return list(audio_dir.glob("*.wav"))
        except OSError:
            sleep(1)
    else:
        raise OSError(f"Failed to list the audio files in {audio_dir!r}.")


############################################
##### Building the conversation subset #####
############################################


# TODO: Implement this function
def build_conversation_dataset(
    metadata_database_path: Path, audio_dir: Path
) -> Dataset:
    """Build the CoRal conversation dataset.

    Args:
        metadata_database_path:
            Path to the SQLite database containing the metadata.
        audio_dir:
            Path to the directory containing the audio files.

    Returns:
        The CoRal conversation dataset.
    """
    dataset = Dataset.from_dict({})
    return dataset


#####################################
##### Splitting of the datasets #####
#####################################


def split_dataset(
    dataset: Dataset, test_speakers: list[str], val_speakers: list[str]
) -> DatasetDict | None:
    """Split a dataset into train, validation and test sets.

    Args:
        dataset:
            The dataset to split.
        test_speakers:
            A list of speakers in the test set.
        val_speakers:
            A list of speakers in the validation set.

    Returns:
        The split dataset, or None if no training samples are found.

    Raises:
        ValueError:
            If no training samples are found.
    """
    if len(dataset) == 0:
        return None

    with no_datasets_progress_bars():
        train_dataset = dataset.filter(
            function=partial(
                examples_belong_to_train,
                test_speakers=test_speakers,
                val_speakers=val_speakers,
            ),
            batched=True,
        )
    splits = dict(train=train_dataset)

    with no_datasets_progress_bars():
        val_dataset = dataset.filter(
            function=partial(examples_belong_to_val, val_speakers=val_speakers),
            batched=True,
        )
    if len(val_dataset) > 0:
        splits["val"] = val_dataset

    with no_datasets_progress_bars():
        test_dataset = dataset.filter(
            function=partial(examples_belong_to_test, test_speakers=test_speakers),
            batched=True,
        )
    if len(test_dataset) > 0:
        splits["test"] = test_dataset

    return DatasetDict(splits)


def examples_belong_to_train(
    examples: dict[str, list], test_speakers: list[str], val_speakers: list[str]
) -> list[bool]:
    """Check if each example belongs to the training set.

    Args:
        examples:
            A batch of examples.
        test_speakers:
            A list of speakers in the test set.
        val_speakers:
            A list of speakers in the validation set.

    Returns:
        A list of booleans indicating whether each example belongs to the training
        set.
    """
    return [
        speaker_id not in test_speakers + val_speakers
        for speaker_id in examples["id_speaker"]
    ]


def examples_belong_to_val(
    examples: dict[str, list], val_speakers: list[str]
) -> list[bool]:
    """Check if each example belongs to the validation set.

    Args:
        examples:
            A batch of examples.
        val_speakers:
            A list of speakers in the validation set.

    Returns:
        A list of booleans indicating whether each example belongs to the validation
        set.
    """
    return [speaker_id in val_speakers for speaker_id in examples["id_speaker"]]


def examples_belong_to_test(
    examples: dict[str, list], test_speakers: list[str]
) -> list[bool]:
    """Check if each example belongs to the test set.

    Args:
        examples:
            A batch of examples.
        test_speakers:
            A list of speakers in the test set.

    Returns:
        A list of booleans indicating whether each example belongs to the test set.
    """
    return [speaker_id in test_speakers for speaker_id in examples["id_speaker"]]


#####################################
##### Uploading of the datasets #####
#####################################


def upload_dataset(
    read_aloud_dataset: DatasetDict | None,
    conversation_dataset: DatasetDict | None,
    hub_id: str,
) -> None:
    """Upload the dataset to the Hugging Face Hub.

    Args:
        read_aloud_dataset:
            The read-aloud dataset, or None if no such dataset exists.
        conversation_dataset:
            The conversation dataset, or None if no such dataset exists.
        hub_id:
            Identifier of the Hugging Face Hub repository.
    """
    if read_aloud_dataset is not None:
        for _ in range(60):
            try:
                read_aloud_dataset.push_to_hub(
                    repo_id=hub_id,
                    config_name="read_aloud",
                    private=True,
                    max_shard_size="500MB",
                    commit_message="Add the CoRal read-aloud dataset",
                )
                break
            except (RuntimeError, HTTPError) as e:
                logger.info(f"Error while pushing to hub: {e}")
                logger.info("Waiting a minute before trying again...")
                sleep(60)
                logger.info("Retrying...")
        else:
            logger.error("Failed to upload the read-aloud dataset.")

    if conversation_dataset is not None:
        for _ in range(60):
            try:
                conversation_dataset.push_to_hub(
                    repo_id=hub_id,
                    config_name="conversation",
                    private=True,
                    max_shard_size="500MB",
                    commit_message="Add the CoRal conversation dataset",
                )
                break
            except (RuntimeError, HTTPError) as e:
                logger.info(f"Error while pushing to hub: {e}")
                logger.info("Waiting a minute before trying again...")
                sleep(60)
                logger.info("Retrying...")
    else:
        logger.error("Failed to upload the conversation dataset.")


#############################
##### Utility functions #####
#############################


def copy_audio_directory_to_cwd(audio_dir: Path) -> Path:
    """Copy audio files to the current working directory.

    Args:
        audio_dir:
            The directory containing the audio files.
        max_attempts (optional):
            The maximum number of attempts to list the audio files. Defaults to 10.

    Returns:
        The new directory containing the audio files.
    """
    new_audio_dir = Path.cwd() / audio_dir.name
    new_audio_dir.mkdir(exist_ok=True)

    # Get list of subdirectories of the audio directory, or abort of none exist
    audio_subdirs = [path for path in audio_dir.iterdir() if path.is_dir()]
    if not audio_subdirs:
        return new_audio_dir

    # Compress all subdirectories that are not already compressed
    with Parallel(n_jobs=mp.cpu_count(), backend="threading") as parallel:
        parallel(
            delayed(function=compress_dir)(directory=subdir)
            for subdir in tqdm(
                iterable=audio_subdirs,
                desc="Compressing audio files on the source disk",
            )
        )

    # Decompress all the compressed audio files in the current working directory
    with Parallel(n_jobs=mp.cpu_count(), backend="threading") as parallel:
        parallel(
            delayed(function=decompress_file)(
                file=compressed_subdir, destination_dir=new_audio_dir
            )
            for compressed_subdir in tqdm(
                iterable=list(audio_dir.glob("*.tar.xz")),
                desc="Copying the compressed files and decompressing them",
            )
        )

    return new_audio_dir


def compress_dir(directory: Path) -> Path:
    """Compress a directory using tar.

    Args:
        directory:
            The directory to compress.

    Returns:
        The path to the compressed file.
    """
    if not directory.with_suffix(".tar.xz").exists():
        with tarfile.open(name=f"{str(directory)}.tar.xz", mode="w:xz") as tar:
            tar.add(name=directory, arcname=directory.name)
    return directory.with_suffix(".tar.xz")


def decompress_file(file: Path, destination_dir: Path) -> None:
    """Decompress a tarfile into a directory.

    Args:
        file:
            The file to decompress.
        destination_dir:
            The destination directory.
    """
    destination_path = destination_dir / file.name
    decompressed_path = remove_suffixes(path=destination_path)
    if not decompressed_path.exists():
        if not destination_path.exists():
            shutil.copy(src=file, dst=destination_dir)
        try:
            with tarfile.open(name=destination_path, mode="r:xz") as tar:
                tar.extractall(path=destination_dir)
        except Exception as e:
            logging.error(
                f"Failed to decompress the file {file} - it appears to be corrupted. "
                f"The error message was: {e}"
            )
            shutil.rmtree(decompressed_path, ignore_errors=True)
            file.unlink()
        destination_path.unlink()


def remove_suffixes(path: Path) -> Path:
    """Remove all suffixes from a path, even if it has multiple.

    Args:
        path:
            The path to remove the suffixes from.

    Returns:
        The path without any suffixes.
    """
    while path.suffix:
        path = path.with_suffix("")
    return path


class CorruptedCompressedFile(Exception):
    """Exception raised when a compressed file is corrupted."""

    def __init__(self, file: Path) -> None:
        """Initialise the exception.

        Args:
            file:
                The corrupted file.
        """
        self.file = file
        self.message = (
            f"Failed to decompress the file {self.file}, as it appears to be corrupted."
        )
        super().__init__(self.message)


if __name__ == "__main__":
    main()
