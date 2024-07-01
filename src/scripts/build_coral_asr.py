"""Script that builds and uploads the CoRal speech recognition dataset from the raw data.

Usage:
    python src/scripts/build_coral_asr.py --nas-dir <path/to/nas/dir> --hub-id <hub-id>
"""

import logging
import sqlite3
from pathlib import Path

import click
from datasets import Dataset

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
def main(
    audio_dir: Path | str, metadata_database_path: Path | str, hub_id: str
) -> None:
    """Build and upload the CoRal speech recognition dataset."""
    logger.info("Initialised the build script of the CoRal speech recognition dataset.")

    metadata_database_path = Path(metadata_database_path)
    audio_dir = Path(audio_dir)

    non_id_features = [
        "text",
        "datetime_start",
        "datetime_end",
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
        LIMIT 10
    """

    logger.info("Fetching all metadata from the SQLite database...")
    with sqlite3.connect(database=metadata_database_path) as connection:
        cursor = connection.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()

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
        }
    )

    logger.info(f"Fetched {len(dataset):,} rows from the SQLite database.")
    logger.info("Fetching all corresponding audio files...")

    # TODO: Implement the fetching of the audio files.

    logger.info("Finished fetching all audio files. Building the dataset...")

    # TODO: Implement the building of the dataset.

    logger.info(
        f"Finished building the dataset. Uploading the dataset to {hub_id!r} on "
        "the Hugging Face Hub..."
    )

    # TODO: Implement the uploading of the dataset to the Hugging Face Hub.

    logger.info("Finished uploading the dataset to the Hugging Face Hub. All done!")


if __name__ == "__main__":
    main()
