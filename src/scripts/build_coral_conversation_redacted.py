"""Build the CoRal redacted conversation dataset.

Removes personal information from audio files based on transcription annotations,
replaces speaker names with anonymized IDs, and stores the result as a
HuggingFace dataset (locally, with optional upload).
"""

import logging
import shutil
import sqlite3
from pathlib import Path
from time import sleep
from typing import Any

import hydra
import pysubs2
from datasets import Audio, Dataset, Features, Sequence, Value
from omegaconf import DictConfig
from progress.bar import IncrementalBar
from pydub import AudioSegment
from requests import HTTPError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_coral_conversation_redacted")


@hydra.main(
    config_path="../../config",
    config_name="dataset_creation_conversation_redacted",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Build the redacted CoRal conversation dataset.

    Args:
        config:
            The Hydra configuration object.
    """
    metadata_database_path = Path(config.metadata_database_path)
    audio_dir = Path(config.audio_dir)
    transcript_dir = Path(config.transcripts_dir)
    pi_marker = config.personal_information

    output_dir = Path(config.new_local_dir)
    new_audio_dir = output_dir / audio_dir.name
    new_audio_dir.mkdir(exist_ok=True, parents=True)
    new_transcript_dir = output_dir / transcript_dir.name
    new_transcript_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Building the CoRal redacted conversation dataset...")
    dataset = build_redacted_conversation_dataset(
        metadata_database_path=metadata_database_path,
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        new_audio_dir=new_audio_dir,
        new_transcript_dir=new_transcript_dir,
        pi_marker=pi_marker,
    )

    if dataset is None:
        logger.error("No conversations were processed. Exiting.")
        return

    dataset_path = output_dir / "dataset"
    logger.info(f"Saving dataset to {dataset_path}")
    dataset.save_to_disk(str(dataset_path))

    if config.get("upload", False):
        upload_dataset(dataset, hub_id=config.hub_id)


def build_redacted_conversation_dataset(
    metadata_database_path: Path,
    audio_dir: Path,
    transcript_dir: Path,
    new_audio_dir: Path,
    new_transcript_dir: Path,
    pi_marker: str,
) -> Dataset | None:
    """Build the redacted CoRal conversation dataset.

    Args:
        metadata_database_path:
            Path to the SQLite database containing the metadata.
        audio_dir:
            Directory containing the audio files.
        transcript_dir:
            Directory containing the transcription files (.ass format).
        new_audio_dir:
            Directory to store the redacted audio files.
        new_transcript_dir:
            Directory to store the modified transcription files.
        pi_marker:
            The marker string indicating personal information in transcriptions.

    Returns:
        The redacted conversation dataset, or None if no conversations matched.
    """
    conversation_rows, speaker_rows = extract_metadata(metadata_database_path)

    all_transcription_files = {p.stem: p for p in transcript_dir.glob("*.ass")}
    logger.info(f"Found {len(all_transcription_files)} transcription files")

    matchable = [
        (conv_id, conv_meta)
        for conv_id, conv_meta in conversation_rows.items()
        if conv_id in all_transcription_files
        and find_audio_file(audio_dir, conv_id) is not None
    ]
    logger.info(
        f"Matched {len(matchable)} conversations with both audio and transcription"
    )

    if not matchable:
        return None

    dataset_rows: list[dict[str, Any]] = []

    with IncrementalBar(
        "Processing conversations",
        max=len(matchable),
        suffix="%(index)d/%(max)d [%(eta_td)s / %(elapsed_td)s]",
    ) as bar:
        for conv_id, conv_meta in matchable:
            transcript_path = all_transcription_files[conv_id]
            audio_path = find_audio_file(audio_dir, conv_id)

            try:
                transcription = pysubs2.load(str(transcript_path))
            except Exception as e:
                logger.warning(f"Failed to parse {transcript_path}: {e}")
                bar.next()
                continue

            speaker_map = {
                "A": conv_meta["id_speaker_a"],
                "B": conv_meta["id_speaker_b"],
                "C": conv_meta["id_recorder"],
            }

            segments, pi_intervals = process_transcription(
                transcription, speaker_map, pi_marker
            )

            new_audio_path = redact_and_store_audio(
                audio_path, pi_intervals, new_audio_dir
            )
            store_modified_transcript(
                transcription, transcript_path.name, new_transcript_dir
            )

            if new_audio_path is None:
                bar.next()
                continue

            row: dict[str, Any] = {
                "id_conversation": conv_id,
                "location": conv_meta.get("location"),
                "location_roomdim": conv_meta.get("location_roomdim"),
                "noise_level": conv_meta.get("noise_level"),
                "noise_type": conv_meta.get("noise_type"),
                "audio": str(new_audio_path),
                "transcription_segments": segments,
            }

            for role, key in [
                ("speaker_a", "id_speaker_a"),
                ("speaker_b", "id_speaker_b"),
                ("recorder", "id_recorder"),
            ]:
                speaker_id = conv_meta[key]
                speaker = speaker_rows.get(speaker_id, {})
                row[f"{role}_id"] = speaker_id
                row[f"{role}_age"] = speaker.get("age")
                row[f"{role}_gender"] = speaker.get("gender")
                row[f"{role}_dialect"] = speaker.get("dialect")
                row[f"{role}_country_birth"] = speaker.get("country_birth")
                row[f"{role}_education"] = speaker.get("education")
                row[f"{role}_occupation"] = speaker.get("occupation")

            dataset_rows.append(row)
            bar.next()

    logger.info(f"Built dataset with {len(dataset_rows)} conversations")

    if not dataset_rows:
        return None

    features = Features(
        {
            "id_conversation": Value("string"),
            "location": Value("string"),
            "location_roomdim": Value("string"),
            "noise_level": Value("string"),
            "noise_type": Value("string"),
            "audio": Audio(),
            "transcription_segments": Sequence(
                {
                    "speaker_id": Value("string"),
                    "start_seconds": Value("float64"),
                    "end_seconds": Value("float64"),
                    "text": Value("string"),
                    "is_redacted": Value("bool"),
                }
            ),
            "speaker_a_id": Value("string"),
            "speaker_a_age": Value("string"),
            "speaker_a_gender": Value("string"),
            "speaker_a_dialect": Value("string"),
            "speaker_a_country_birth": Value("string"),
            "speaker_a_education": Value("string"),
            "speaker_a_occupation": Value("string"),
            "speaker_b_id": Value("string"),
            "speaker_b_age": Value("string"),
            "speaker_b_gender": Value("string"),
            "speaker_b_dialect": Value("string"),
            "speaker_b_country_birth": Value("string"),
            "speaker_b_education": Value("string"),
            "speaker_b_occupation": Value("string"),
            "recorder_id": Value("string"),
            "recorder_age": Value("string"),
            "recorder_gender": Value("string"),
            "recorder_dialect": Value("string"),
            "recorder_country_birth": Value("string"),
            "recorder_education": Value("string"),
            "recorder_occupation": Value("string"),
        }
    )

    return Dataset.from_dict(
        {key: [row[key] for row in dataset_rows] for key in dataset_rows[0]},
        features=features,
    )


def extract_metadata(
    metadata_database_path: Path,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Extract conversation and speaker metadata from the SQLite database.

    Args:
        metadata_database_path:
            Path to the SQLite database.

    Returns:
        A tuple of (conversation_rows, speaker_rows) where conversation_rows maps
        conversation IDs to their metadata, and speaker_rows maps speaker IDs to
        their demographic data.
    """
    with sqlite3.connect(database=metadata_database_path) as connection:
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM Conversations")
        count = cursor.fetchone()[0]
        logger.info(f"There are {count:,} conversations in the database.")

        cursor.execute(
            """
            SELECT
                id_conversation,
                id_speaker_a,
                id_speaker_b,
                id_recorder,
                location,
                location_roomdim,
                noise_level,
                noise_type
            FROM Conversations
            """
        )
        conversation_rows = {
            row[0]: {
                "id_speaker_a": row[1],
                "id_speaker_b": row[2],
                "id_recorder": row[3],
                "location": row[4],
                "location_roomdim": row[5],
                "noise_level": row[6],
                "noise_type": row[7],
            }
            for row in cursor.fetchall()
        }

        cursor.execute(
            """
            SELECT
                id_speaker, age, gender, dialect,
                country_birth, education, occupation
            FROM Speakers
            WHERE id_speaker IN (
                SELECT id_speaker_a FROM Conversations UNION
                SELECT id_speaker_b FROM Conversations UNION
                SELECT id_recorder FROM Conversations
            )
            """
        )
        speaker_rows = {
            row[0]: {
                "age": row[1],
                "gender": row[2],
                "dialect": row[3],
                "country_birth": row[4],
                "education": row[5],
                "occupation": row[6],
            }
            for row in cursor.fetchall()
        }

    logger.info(
        f"Got {len(conversation_rows)} conversations with "
        f"{len(speaker_rows)} distinct speakers"
    )
    return conversation_rows, speaker_rows


def process_transcription(
    transcription: pysubs2.SSAFile,
    speaker_map: dict[str, str],
    pi_marker: str,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Process a transcription: anonymize speakers and identify PI intervals.

    Args:
        transcription:
            The parsed transcription (modified in place for speaker names).
        speaker_map:
            Mapping of original speaker labels (A, B, C) to anonymized IDs.
        pi_marker:
            The marker string indicating personal information.

    Returns:
        A tuple of (segments, pi_intervals) where segments is a list of dicts
        for the dataset and pi_intervals is a list of (start_ms, end_ms) tuples.
    """
    segments: list[dict] = []
    pi_intervals: list[tuple[int, int]] = []

    for event in transcription:
        speaker_id = speaker_map.get(event.name, event.name)
        event.name = speaker_id

        is_redacted = pi_marker in event.text
        if is_redacted:
            event.text = pi_marker
            pi_intervals.append((event.start, event.end))

        segments.append(
            {
                "speaker_id": speaker_id,
                "start_seconds": event.start / 1000.0,
                "end_seconds": event.end / 1000.0,
                "text": event.text,
                "is_redacted": is_redacted,
            }
        )

    return segments, pi_intervals


def redact_and_store_audio(
    audio_path: Path,
    pi_intervals: list[tuple[int, int]],
    new_audio_dir: Path,
) -> Path | None:
    """Redact PI intervals from audio by replacing them with silence.

    Args:
        audio_path:
            Path to the original audio file.
        pi_intervals:
            List of (start_ms, end_ms) tuples to silence.
        new_audio_dir:
            Directory to store the output audio.

    Returns:
        Path to the new audio file, or None if processing fails.
    """
    try:
        new_path = new_audio_dir / f"{audio_path.stem}.wav"
        if pi_intervals:
            audio = AudioSegment.from_file(audio_path)
            for start_ms, end_ms in pi_intervals:
                audio = (
                    audio[:start_ms]
                    + AudioSegment.silent(duration=(end_ms - start_ms))
                    + audio[end_ms:]
                )
            audio.export(new_path, format="wav")
        else:
            if audio_path.suffix == ".wav":
                shutil.copy(audio_path, new_path)
            else:
                audio = AudioSegment.from_file(audio_path)
                audio.export(new_path, format="wav")
        return new_path
    except Exception as e:
        logger.warning(f"Failed to process audio file {audio_path}: {e}")
        return None


def store_modified_transcript(
    transcription: pysubs2.SSAFile,
    filename: str,
    new_transcript_dir: Path,
) -> None:
    """Write the modified transcription (.ass) to disk for manual inspection.

    Args:
        transcription:
            The modified transcription with anonymized speaker names.
        filename:
            Original filename to preserve.
        new_transcript_dir:
            Directory to write the file to.
    """
    output_path = new_transcript_dir / filename
    transcription.save(str(output_path))


def find_audio_file(audio_dir: Path, name: str) -> Path | None:
    """Find an audio file by name in the specified directory.

    Args:
        audio_dir:
            Directory containing the audio files.
        name:
            Name of the audio file without extension.

    Returns:
        Path to the audio file if found, otherwise None.
    """
    for ext in ("wav", "m4a"):
        path = audio_dir / f"{name}.{ext}"
        if path.exists():
            return path
    logger.warning(f"No audio file found for {name}")
    return None


def upload_dataset(dataset: Dataset, hub_id: str) -> None:
    """Upload the dataset to the HuggingFace Hub with retry logic.

    Args:
        dataset:
            The dataset to upload.
        hub_id:
            HuggingFace Hub repository ID.
    """
    logger.info(f"Uploading dataset to {hub_id}...")
    for _ in range(60):
        try:
            dataset.push_to_hub(
                repo_id=hub_id,
                config_name="v3_full_conversation_redacted",
                private=True,
                max_shard_size="500MB",
                commit_message="Add the CoRal full conversations redacted dataset",
            )
            logger.info("Upload complete.")
            return
        except (RuntimeError, HTTPError) as e:
            logger.info(f"Error while pushing to hub: {e}")
            logger.info("Waiting a minute before trying again...")
            sleep(60)
    logger.error("Failed to upload the redacted conversation dataset.")


if __name__ == "__main__":
    main()
