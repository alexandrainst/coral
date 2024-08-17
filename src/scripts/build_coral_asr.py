"""Script that builds and uploads the CoRal speech recognition dataset from the raw data.

Usage:
    python src/scripts/build_coral_asr.py \
        [--audio-dir directory/containing/the/audio/subdirectories] \
        [--metadata-database-path path/to/the/sqlite/database] \
        [--hub-id organisation/dataset-id]
"""

import logging
import multiprocessing as mp
import shutil
import sqlite3
import tarfile
from pathlib import Path
from time import sleep

import click
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    disable_progress_bar,
    enable_progress_bar,
)
from joblib import Parallel, delayed
from requests import HTTPError
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_coral_asr")


# Estimated number of hours: 7.50
# Gender distribution:
# - female: 57%
# - male: 43%
# Dialect distribution:
# - Bornholmsk: 10%
# - Fynsk: 11%
# - Københavnsk: 12%
# - Nordjysk: 11%
# - Sjællandsk: 11%
# - Sydømål: 11%
# - Sønderjysk: 12%
# - Vestjysk: 11%
# - Østjysk: 12%
# Age_group distribution:
# - 0-24: 31%
# - 25-49: 35%
# - 50-: 33%
# Accent distribution:
# - native: 80%
# - foreign: 20%
TEST_SET_SPEAKER_IDS: list[str] = [
    "spe_fe46df2f5904e1443590a5aa735946ca",
    "spe_10b8eb8c3ba5fd8405d1516b7b12f2de",
    "spe_65388692c15793f7746fe454931bdf49",
    "spe_7b8398c898a828791c0fc40d6d146b3f",
    "spe_a3d4edeab8ea4c9bb67e847103c4b5f7",
    "spe_7902a69e777bf0fa15ae75c7ce913503",
    "spe_3dd62e87b39a71dc50aaf90199dad34b",
    "spe_55028d05581a88a8655fa1f74ddfb5a1",
    "spe_035a5bcd2f0670c021e8c70d4effc563",
    "spe_3aba2e587b07b3bc7ecb121bee547360",
    "spe_4e8a1be0def20faf50d02a084a9adb6c",
    "spe_434d5c8eb40fd915bdae28bc590d829c",
    "spe_af4e767c077909a95b9bd834ca224833",
    "spe_b977ebc0a2ba961cbe158190fce0dc06",
    "spe_5f548d16aa39584dfd775c3a80b404a7",
    "spe_b31ef7d6e97a7fd7ce8bf169600c64a2",
    "spe_44dd789a42cfb357b3cd0145f667ffda",
    "spe_0d673a6ff07fd1895ad0be7714755c78",
    "spe_ae8bb53db7e325a8ecbb3238f4578d38",
    "spe_b96ee10564409029a6810562e34cd7a1",
    "spe_f3a0b2f9a75fcfc793a3109d8fbd6c94",
    "spe_647d4e905427d45ab699abe73d80ef1d",
    "spe_6617b4c7273b31fc161fc6e07e620743",
    "spe_9b021d63f84de498fc97b75367678e78",
    "spe_23c3eb37310f6ae61b3b275e88157309",
    "spe_e01017cbabe39aa19980d30b022947dc",
    "spe_6cb7a0be3ea54ae06b2c5c0c5d349347",
    "spe_497254e1a7f3b8235252224fba53461f",
    "spe_8948a0cc310c6fa8161665d4eda79846",
    "spe_6e67cbe51a49d9e4abbd7699a4a89d91",
    "spe_f5bb05a736c91f4347edc51e4199278a",
    "spe_e3742811d83011e22ec2ef5a7af32065",
    "spe_20b91d51f72ee56930ca778cb16c29da",
]

# Estimated number of hours: 1.43
# Gender distribution:
# - female: 52%
# - male: 48%
# Dialect distribution:
# - Bornholmsk: 13%
# - Fynsk: 8%
# - Københavnsk: 7%
# - Nordjysk: 5%
# - Sjællandsk: 8%
# - Sydømål: 17%
# - Sønderjysk: 14%
# - Vestjysk: 16%
# - Østjysk: 12%
# Age_group distribution:
# - 0-24: 21%
# - 25-49: 36%
# - 50-: 42%
# Accent distribution:
# - native: 94%
# - foreign: 6%
VALIDATION_SET_SPEAKER_IDS: list[str] = [
    "spe_d5ec9dc47d76e3be1bc451561c6cf655",
    "spe_4aa23a60464a18e3597cdeb3606ac572",
    "spe_b9112f9327f2390093bbc082a1651bad",
    "spe_b8669c732e901851c13ef1ee7f138e48",
    "spe_6aeb15b456086536f45918dbdfc63ec6",
    "spe_40ca0d47aa6dfd99e56f2afdca4b3ee9",
    "spe_c3c1fdae39d6bf6e462868f8f52b7e3e",
    "spe_ced5114cc6dc923dcb1bcc3db3480691",
    "spe_e3013f96eed48bacc13dd8253609cf9b",
    "spe_f64c7781f78d1ac24a979acb3080a1d6",
    "spe_6e7cb65603907f863e06d7a02e00fb67",
]


@click.command()
@click.option(
    "--audio-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default="/Volumes/CoRal/_new_structure/raw",
    show_default=True,
    help="Path to the directory containing the raw audio files.",
)
@click.option(
    "--metadata-database-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="/Volumes/CoRal/_new_structure/raw/CoRal_public.db",
    show_default=True,
    help="Path to the SQLite database containing the metadata.",
)
@click.option(
    "--hub-id",
    type=str,
    default="alexandrainst/coral",
    show_default=True,
    help="Identifier of the Hugging Face Hub repository.",
)
def main(
    audio_dir: Path | str, metadata_database_path: Path | str, hub_id: str
) -> None:
    """Build and upload the CoRal speech recognition dataset."""
    metadata_database_path = Path(metadata_database_path)
    read_aloud_dir = Path(audio_dir) / "recordings"
    conversation_dir = Path(audio_dir) / "conversations"

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

    logger.info("Splitting the datasets into train, validation and test sets...")
    read_aloud_dataset = split_dataset(dataset=read_aloud_dataset)
    conversation_dataset = split_dataset(dataset=conversation_dataset)

    logger.info(f"Uploading the datasets to {hub_id!r} on the Hugging Face Hub...")
    upload_dataset(
        read_aloud_dataset=read_aloud_dataset,
        conversation_dataset=conversation_dataset,
        hub_id=hub_id,
    )

    logger.info(f"All done! See the datasets at https://hf.co/datasets/{hub_id}.")


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


def split_dataset(dataset: Dataset) -> DatasetDict | None:
    """Split a dataset into train, validation and test sets.

    Args:
        dataset:
            The dataset to split.

    Returns:
        The split dataset, or None if no training samples are found.

    Raises:
        ValueError:
            If no training samples are found.
    """
    if len(dataset) == 0:
        return None

    with no_progress_bar():
        train_dataset = dataset.filter(function=examples_belong_to_train, batched=True)
    splits = dict(train=train_dataset)

    with no_progress_bar():
        val_dataset = dataset.filter(function=examples_belong_to_val, batched=True)
    if len(val_dataset) > 0:
        splits["val"] = val_dataset

    with no_progress_bar():
        test_dataset = dataset.filter(function=examples_belong_to_test, batched=True)
    if len(test_dataset) > 0:
        splits["test"] = test_dataset

    return DatasetDict(splits)


def examples_belong_to_train(examples: dict[str, list]) -> list[bool]:
    """Check if each example belongs to the training set.

    Args:
        examples:
            A batch of examples.

    Returns:
        A list of booleans indicating whether each example belongs to the training
        set.
    """
    return [
        speaker_id not in VALIDATION_SET_SPEAKER_IDS + TEST_SET_SPEAKER_IDS
        for speaker_id in examples["id_speaker"]
    ]


def examples_belong_to_val(examples: dict[str, list]) -> list[bool]:
    """Check if each example belongs to the validation set.

    Args:
        examples:
            A batch of examples.

    Returns:
        A list of booleans indicating whether each example belongs to the validation
        set.
    """
    return [
        speaker_id in VALIDATION_SET_SPEAKER_IDS
        for speaker_id in examples["id_speaker"]
    ]


def examples_belong_to_test(examples: dict[str, list]) -> list[bool]:
    """Check if each example belongs to the test set.

    Args:
        examples:
            A batch of examples.

    Returns:
        A list of booleans indicating whether each example belongs to the test set.
    """
    return [speaker_id in TEST_SET_SPEAKER_IDS for speaker_id in examples["id_speaker"]]


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


class no_progress_bar:
    """Context manager that disables the progress bar."""

    def __enter__(self):
        """Disable the progress bar."""
        disable_progress_bar()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Re-enable the progress bar."""
        enable_progress_bar()


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
