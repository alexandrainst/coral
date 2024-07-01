"""Script that builds and uploads the CoRal speech recognition dataset from the raw data.

Usage:
    python src/scripts/build_coral_asr.py \
        [--audio-dir <audio_dir>] \
        [--metadata-database-path <metadata_database_path>] \
        [--hub-id <hub_id>] \
        [--batch-size <batch_size>]
"""

import logging
import sqlite3
from pathlib import Path

import click
from datasets import Audio, Dataset
from joblib import Parallel, delayed
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_coral_asr")


@click.command()
@click.option(
    "--audio-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default="/Volumes/CoRal/_new_structure/raw/recordings",
    show_default=True,
    help="Relative path to the directory containing the audio files.",
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
@click.option(
    "--batch-size",
    type=int,
    default=1000,
    show_default=True,
    help="Number of rows to fetch from the SQLite database at once.",
)
def main(
    audio_dir: Path | str,
    metadata_database_path: Path | str,
    hub_id: str,
    batch_size: int,
) -> None:
    """Build and upload the CoRal speech recognition dataset."""
    logger.info("Initialised the build script of the CoRal speech recognition dataset.")

    metadata_database_path = Path(metadata_database_path)
    audio_dir = Path(audio_dir)

    # Get the number of samples in the SQLite database. We don't do any merges here to
    # save some time. That means that the count will be an upper bound rather than a
    # precise number of samples, but we deal with that when we actually fetch the data
    logger.info("Fetching the number of metadata samples in the SQLite database...")
    count_query = "SELECT COUNT(*) FROM Recordings"
    with sqlite3.connect(database=metadata_database_path) as connection:
        cursor = connection.cursor()
        cursor.execute(count_query)
        num_metadata_samples = cursor.fetchone()[0]
    logger.info(f"There are {num_metadata_samples:,} samples in the SQLite database.")

    # Fetch all metadata from the SQLite database
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
    sql_query = f"""
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
        LIMIT {batch_size}
        OFFSET {{offset}}
    """
    num_batches = num_metadata_samples // batch_size
    if num_metadata_samples % batch_size:
        num_batches += 1
    rows: list[list[str]] = list()
    with tqdm(total=num_metadata_samples, desc="Fetching metadata") as pbar:
        for batch_idx in range(num_batches):
            with sqlite3.connect(database=metadata_database_path) as connection:
                cursor = connection.cursor()
                cursor.execute(sql_query.format(offset=batch_idx * batch_size))
                batch_rows = list(map(list, cursor.fetchall()))
            if not batch_rows:
                break
            rows.extend(batch_rows)
            pbar.update(len(batch_rows))
        pbar.update(num_metadata_samples - pbar.n)

    # Get a list of all the audio file paths. We need this since the audio files lie in
    # subdirectories of the main audio directory
    audio_subdirs = list(audio_dir.iterdir())
    with Parallel(n_jobs=-1) as parallel:
        all_audio_path_lists = parallel(
            delayed(list_audio_files)(subdir)
            for subdir in tqdm(audio_subdirs, desc="Collecting audio file paths")
        )
        assert isinstance(all_audio_path_lists, list)
    all_audio_paths = [
        path for path_list in all_audio_path_lists for path in path_list or []
    ]

    # Match the audio files to the metadata, to ensure that there is a 1-to-1
    # correspondence between them
    row_ids_to_keep: list[int] = list()
    for idx, row in enumerate(tqdm(rows, desc="Matching audio files to metadata")):
        recording_id: str = row[0]
        candidate_audio_paths = [
            path for path in all_audio_paths if recording_id in path.stem
        ]
        if not candidate_audio_paths:
            continue
        row.append(str(candidate_audio_paths[0]))
        row_ids_to_keep.append(idx)
    rows = [row for idx, row in enumerate(rows) if idx in row_ids_to_keep]

    # Build the dataset from the metadata and the audio files. This embeds all the audio
    # files into the dataset as parquet files
    logger.info("Building the dataset...")
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
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    logger.info("Finished building the dataset.")

    # Upload the dataset to the Hugging Face Hub
    logger.info(f"Uploading the dataset to {hub_id!r} on the Hugging Face Hub...")
    dataset.push_to_hub(
        repo_id=hub_id, split="train", private=True, max_shard_size="500MB"
    )
    logger.info("Finished uploading the dataset to the Hugging Face Hub.")

    logger.info(f"All done! See the dataset at https://hf.co/datasets/{hub_id}.")


def list_audio_files(audio_dir: Path) -> list[Path]:
    """List all the audio files in the given directory.

    Args:
        audio_dir:
            The directory containing the audio files.

    Returns:
        A list of paths to the audio files.
    """
    return list(audio_dir.glob("*.wav"))


if __name__ == "__main__":
    main()
