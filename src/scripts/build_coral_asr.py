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
# - female: 58%
# - male: 42%
# Dialect distribution:
# - Bornholmsk: 11%
# - Fynsk: 12%
# - Københavnsk: 11%
# - Nordjysk: 10%
# - Sjællandsk: 10%
# - Sydømål: 10%
# - Sønderjysk: 13%
# - Vestjysk: 10%
# - Østjysk: 12%
# Age_group distribution:
# - 0-24: 34%
# - 25-49: 33%
# - 50-: 33%
# Accent distribution:
# - native: 87%
# - foreign: 13%
TEST_SET_SPEAKER_IDS: list[str] = [
    "spe_c3c1fdae39d6bf6e462868f8f52b7e3e",
    "spe_d7da3d62a1a9885c1cb3280668437759",
    "spe_ce5c35bd408b296511ce0b05ecc33de1",
    "spe_e3013f96eed48bacc13dd8253609cf9b",
    "spe_4facb80c94341b25425ec1d8962b1f8d",
    "spe_20b91d51f72ee56930ca778cb16c29da",
    "spe_e33e46611f54ae91ed7b235c11ef2628",
    "spe_6b93da4530e853772df0fc8b2337142c",
    "spe_6617b4c7273b31fc161fc6e07e620743",
    "spe_1c32c35e35670f64d9e1a673c47aabd1",
    "spe_590765d7656376e83a33c54f9c2e3976",
    "spe_066938532ac270d527696f89d81f0de4",
    "spe_545558c5701d956a2c63057cd313ff50",
    "spe_af4e767c077909a95b9bd834ca224833",
    "spe_4b7ba1403d8540b3101c07b9c8a19474",
    "spe_e3742811d83011e22ec2ef5a7af32065",
    "spe_fbf3381f525dbe5ddf1a2a1d36e9c4b9",
    "spe_ef7f083c2097793e28388535a81e14ea",
    "spe_b9112f9327f2390093bbc082a1651bad",
    "spe_b788006083ced2efabc75a7907220250",
    "spe_6e7cb65603907f863e06d7a02e00fb67",
    "spe_003c825c9ad2f1496c22cc16a04b1598",
    "spe_935a99ce745c2c042a77f6e4c831fd94",
    "spe_55028d05581a88a8655fa1f74ddfb5a1",
    "spe_6efa16af1112af15ab482171e156d3f3",
    "spe_01fc2b156c7fe429f1b72bd3be5ad3c3",
    "spe_b19c9900784bdf8f8ef3ea3e78002011",
    "spe_02d28146c013111766f18f0d2198785e",
    "spe_3937cb6805e15326b37253d6148babb5",
    "spe_492647a87720047b55f4033d0df8082a",
    "spe_9b8d26599c6b7932dbac00832b73dcf8",
    "spe_6a029298b9eaa3d7e7f8f74510f88e70",
]

# Estimated number of hours: 1.54
# Gender distribution:
# - female: 48%
# - male: 52%
# Dialect distribution:
# - Bornholmsk: 5%
# - Fynsk: 11%
# - Københavnsk: 17%
# - Nordjysk: 9%
# - Sjællandsk: 6%
# - Sydømål: 18%
# - Sønderjysk: 5%
# - Vestjysk: 8%
# - Østjysk: 19%
# Age_group distribution:
# - 0-24: 39%
# - 25-49: 37%
# - 50-: 24%
# Accent distribution:
# - native: 91%
# - foreign: 9%
VALIDATION_SET_SPEAKER_IDS: list[str] = [
    "spe_a8ffed9a90c0e89338892f23dfaac338",
    "spe_50d0664744aa2a241c084363b04e39c5",
    "spe_003971defa823a2eb98f079cdc91c634",
    "spe_0dd042aee46edc27b2ba0155abdf3d54",
    "spe_cd5174de19523f69dd8613ea311997d4",
    "spe_0cf8566481d05e4ed329e34211a36311",
    "spe_4aa23a60464a18e3597cdeb3606ac572",
    "spe_647d4e905427d45ab699abe73d80ef1d",
    "spe_bb1d0e9d3f2bca18658975b3073924cb",
    "spe_93f1d99433d997beeec289d60e074ed2",
    "spe_9c4dc6be57f6c63860331813a71417e5",
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
