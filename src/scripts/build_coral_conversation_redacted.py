"""This script builds the CoRal conversation dataset without splitting of audio.

It removes personal information from the audio files where based on the transcription. It may be modified to upload directly to hugginface if wanted.
"""

import logging
import shutil
import sqlite3
from pathlib import Path
from typing import TypedDict

import ass
import hydra
from ass import Document
from datasets import Dataset
from omegaconf import DictConfig
from pydub import AudioSegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_coral_asr")


@hydra.main(
    config_path="../../config",
    config_name="dataset_creation_conversation_redacted",
    version_base=None,
)
def main(config: DictConfig) -> None:  # noqa: D103
    metadata_database_path = Path(config.metadata_database_path)
    conversation_dir = Path(config.audio_dir)
    trabscript_dir = Path(config.transcripts_dir)

    new_audio_dir = Path.cwd() / Path(config.new_local_dir).name / conversation_dir.name
    new_audio_dir.mkdir(exist_ok=True, parents=True)
    new_transcription_dir = (
        Path.cwd() / Path(config.new_local_dir).name / trabscript_dir.name
    )
    new_transcription_dir.mkdir(exist_ok=True, parents=True)

    # The number of files is small enough to not have to move it locally without noticable performance impac
    logger.info("Building the CoRal read-aloud speech recognition dataset...")
    read_aloud_dataset = build_redacted_conversation_dataset(
        metadata_database_path=metadata_database_path,
        audio_dir=conversation_dir,
        transcription_dir=trabscript_dir,
        new_audio_dir=new_audio_dir,
        new_transcription_dir=new_transcription_dir,
    )


##############################################################
########## Build the conversation dataset
##############################################################


def build_redacted_conversation_dataset(
    metadata_database_path: Path,
    audio_dir: Path,
    transcription_dir: Path,
    new_audio_dir: Path,
    new_transcription_dir: Path,
) -> Dataset:
    """Build the non-formatted CoRal conversation dataset.

    Args:
        metadata_database_path (Path):
            Path to the SQLite database containing the metadata
        audio_dir (Path):
            Directory containing the audio files
        transcription_dir (Path):
            Directory containing the transcription files
        new_audio_dir (Path):
            Directory to move the cleaned audio files to
        new_transcription_dir (Path):
            Directory to move the cleaned transcription files to

    Returns:
        The conversation dataset
    """
    # Load in the speaker data
    rows = extract_metadata(metadata_database_path)

    # Load all .ass files from the transcription directory
    all_transcription_files = list(transcription_dir.glob("*.ass"))

    # Parse the .ass files using the `ass` library
    transcriptions: dict[Path, Document] = {}
    for transcription_filename in all_transcription_files:
        try:
            with open(transcription_filename, "r", encoding="utf-8-sig") as file:
                transcriptions[transcription_filename] = ass.parse(file)
        except Exception as e:
            logger.warning(f"Failed to parse {transcription_filename}: {e}")

    # Iterate over all transcriptions with their name matching up to the audio name
    new_audio_file_paths, new_transcriptions_file_paths = remove_PI_store_new_files(
        audio_dir, new_audio_dir, new_transcription_dir, rows, transcriptions
    )

    return Dataset.from_dict(
        {"audio": new_audio_file_paths, "transcriptions": new_transcriptions_file_paths}
    )


class SpeakerMetadata(TypedDict):  # noqa: D101
    id_speaker_a: str
    id_speaker_b: str
    id_recorder: str


def extract_metadata(metadata_database_path) -> dict[str, SpeakerMetadata]:
    """Extract metadata from the SQLite database.

    Args:
        metadata_database_path (Path): Path to the SQLite database containing metadata.

    Returns:
        dict[str, SpeakerMetadata]: A dictionary mapping conversation IDs to speaker metadata.
    """
    count_query = "SELECT COUNT(*) FROM Conversations;"
    with sqlite3.connect(database=metadata_database_path) as connection:
        cursor = connection.cursor()
        cursor.execute(count_query)
        num_metadata_samples = cursor.fetchone()[0]
    logger.info(f"There are {num_metadata_samples:,} samples in the SQLite database.")

    selection_query = """
        SELECT
            id_conversation,
            id_speaker_a,
            id_speaker_b,
            id_recorder
        FROM
            Conversations
    """

    # Open the database connection and fetch the data
    logger.info("Fetching the metadata from the SQLite database...")
    with sqlite3.connect(database=metadata_database_path) as connection:
        cursor = connection.cursor()
        cursor.execute(selection_query)
        rows = {
            row[0]: {
                "id_speaker_a": row[1],
                "id_speaker_b": row[2],
                "id_recorder": row[3],
            }
            for row in cursor.fetchall()
        }

    return rows


def remove_PI_store_new_files(
    audio_dir: Path,
    new_audio_dir: Path,
    new_transcription_dir: Path,
    rows: dict[str, SpeakerMetadata],
    transcriptions: dict[Path, Document],
) -> tuple[list[Path], list[Path]]:
    """Remove personally identifiable information (PI) from audio and transcription files.

    Args:
        audio_dir (Path): Directory containing the original audio files.
        new_audio_dir (Path): Directory to store the cleaned audio files.
        new_transcription_dir (Path): Directory to store the cleaned transcription files.
        rows (dict[str, SpeakerMetadata]): Metadata mapping conversation IDs to speaker information.
        transcriptions (dict[Path, Document]): Parsed transcription files.

    Returns:
        tuple[list[Path], list[Path]]: Lists of paths to the cleaned audio and transcription files.
    """
    speaker_map = {"A": "id_speaker_a", "B": "id_speaker_b", "C": "id_recorder"}
    new_audio_file_paths: list[Path] = []
    new_transcriptions_file_paths: list[Path] = []

    for transcription_filename, transcription in transcriptions.items():
        row = rows.get(transcription_filename.stem)
        if not row:
            logger.info(
                f"Transcription file {transcription_filename} not found in database"
            )
            continue

        audio_file_path = find_audio_file(audio_dir, transcription_filename.stem)
        if not audio_file_path:
            continue

        pi_intervals = process_transcription(transcription, row, speaker_map)
        new_audio_file_path, modified_transcription_file_path = (
            process_audio_and_transcription(
                audio_file_path,
                transcription_filename,
                transcription,
                pi_intervals,
                new_audio_dir,
                new_transcription_dir,
            )
        )

        if new_audio_file_path and modified_transcription_file_path:
            new_audio_file_paths.append(new_audio_file_path)
            new_transcriptions_file_paths.append(modified_transcription_file_path)

    return new_audio_file_paths, new_transcriptions_file_paths


def process_transcription(
    transcription: Document, row: SpeakerMetadata, speaker_map: dict[str, str]
) -> list[tuple[float, float]]:
    """Process a transcription to replace speaker names and redact personally identifiable information (PI).

    Args:
        transcription (Document): The transcription document to process.
        row (SpeakerMetadata): Metadata for the speakers in the conversation.
        speaker_map (dict[str, str]): Mapping of speaker identifiers to metadata keys.

    Returns:
        list[tuple[float, float]]: A list of intervals (start, end) in seconds where PI was redacted.
    """
    pi_intervals = []
    for event in transcription.events:
        if event.name in speaker_map:
            event.name = row[speaker_map[event.name]]
        if "***PI***" in event.text:
            event.text = "***PI***"
            pi_intervals.append(
                (event.start.total_seconds(), event.end.total_seconds())
            )
    return pi_intervals


def process_audio_and_transcription(
    audio_file_path: Path,
    transcription_filename: Path,
    transcription: Document,
    pi_intervals: list[tuple[float, float]],
    new_audio_dir: Path,
    new_transcription_dir: Path,
) -> tuple[Path | None, Path | None]:
    """Process audio and transcription files to redact (by setting sound to none in interval) personally identifiable information (PI).

    Args:
        audio_file_path (Path): Path to the original audio file.
        transcription_filename (Path): Path to the transcription file.
        transcription (Document): Parsed transcription document.
        pi_intervals (list[tuple[float, float]]): List of intervals (start, end) in seconds where PI is redacted.
        new_audio_dir (Path): Directory to store the cleaned audio file.
        new_transcription_dir (Path): Directory to store the cleaned transcription file.

    Returns:
        tuple[Path | None, Path | None]: Paths to the cleaned audio and transcription files, or None if processing fails.
    """
    new_audio_file_path = new_audio_dir / audio_file_path.name
    modified_transcription_file_path = (
        new_transcription_dir / transcription_filename.name
    )

    try:
        if pi_intervals:
            logger.info(f"Cleaning file {transcription_filename} at {pi_intervals}")
            audio = AudioSegment.from_file(audio_file_path)
            for start, end in pi_intervals:
                start_ms, end_ms = int(start * 1000), int(end * 1000)
                audio = (
                    audio[:start_ms]
                    + AudioSegment.silent(duration=(end_ms - start_ms))
                    + audio[end_ms:]
                )
            audio.export(new_audio_file_path, format="wav")
            with open(
                modified_transcription_file_path, "w", encoding="utf-8-sig"
            ) as file:
                transcription.dump_file(file)
        else:
            logger.info(f"Copying file {audio_file_path.name}")
            shutil.copy(audio_file_path, new_audio_file_path)
            shutil.copy(transcription_filename, modified_transcription_file_path)
    except Exception as e:
        logger.warning(f"Failed to process audio file {audio_file_path}: {e}")
        return None, None

    return new_audio_file_path, modified_transcription_file_path


def find_audio_file(audio_dir: Path, name: str) -> Path | None:
    """Find an audio file by name in the specified directory.

    Args:
        audio_dir (Path): Directory containing the audio files.
        name (str): Name of the audio file without extension.

    Returns:
        Path | None: Path to the audio file if found, otherwise None.
    """
    wav_file = audio_dir / f"{name}.wav"
    m4a_file = audio_dir / f"{name}.m4a"
    if wav_file.exists():
        return wav_file
    elif m4a_file.exists():
        return m4a_file
    else:
        logger.warning(f"No audio file found for {name}")
        return None


if __name__ == "__main__":
    main()