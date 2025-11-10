"""Script that builds and uploads the CoRal speech recognition dataset from raw data.

Usage:
    python src/scripts/build_coral_asr.py [key=value] [key=value] ...
"""

import logging
import multiprocessing as mp
import re
import shutil
import sqlite3
import tarfile
from functools import partial
from itertools import chain
from pathlib import Path
from time import sleep
from typing import Dict, List, Tuple

import hydra
import pandas as pd
import pysubs2
from datasets import Audio, Dataset, DatasetDict
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pydub import AudioSegment
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
    dir_read_aloud = Path(config.dir_data_raw) / "recordings"
    dir_conversation = Path(config.dir_data_raw) / "conversations"
    dir_transcription = Path(config.dir_data_raw) / "transcriptions"

    temp_metadata_database_path = (
        Path.cwd() / "data" / "raw" / metadata_database_path.name
    )
    shutil.copy(src=metadata_database_path, dst=temp_metadata_database_path)

    if config.build_read_aloud:
        logger.info("Building the CoRal read-aloud speech recognition dataset...")
        logger.debug("Copying read-aloud audio files...")
        if config.download2disk:
            dir_read_aloud = copy_audio_directory_to_cwd(audio_dir=dir_read_aloud)

        logger.debug("Building the read-aloud dataset...")
        read_aloud_dataset = build_read_aloud_dataset(
            metadata_database_path=temp_metadata_database_path,
            audio_dir=dir_read_aloud,
            additional_logging=config.debug_logging,
        )

        logger.debug("Validating the read-aloud dataset...")
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

        logger.debug("Splitting the read-aloud dataset...")
        read_aloud_dataset = split_dataset(
            dataset=read_aloud_dataset,
            test_speakers=config.test_speakers,
            val_speakers=config.val_speakers,
        )
    else:
        read_aloud_dataset = None

    if config.build_conversation:
        logger.info("Building the CoRal conversation speech recognition dataset...")

        logger.debug("Copying conversation and transcription files...")
        if config.download2disk:
            dir_conversation = copy_files_to_cwd(source_dir=dir_conversation)
            dir_transcription = copy_files_to_cwd(source_dir=dir_transcription)

        logger.debug("Building the conversation dataset...")
        conversation_dataset = build_conversation_dataset(
            metadata_database_path=temp_metadata_database_path,
            audio_dir=dir_conversation,
            transcript_dir=dir_transcription,
            additional_logging=config.debug_logging,
        )

        logger.debug("Splitting the conversation dataset...")
        conversation_dataset = split_dataset(
            dataset=conversation_dataset,
            test_speakers=config.test_speakers,
            val_speakers=config.val_speakers,
        )
    else:
        conversation_dataset = None

    if config.save2disk:
        logger.info("Saving the datasets to local disk...")
        save_dataset(
            read_aloud_dataset=read_aloud_dataset,
            conversation_dataset=conversation_dataset,
            dir_dest=config.dir_save,
        )

    if config.upload_to_hub:
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


def build_read_aloud_dataset(
    metadata_database_path: Path, audio_dir: Path, additional_logging: bool = False
) -> Dataset:
    """Build the CoRal read-aloud dataset.

    Args:
        metadata_database_path:
            Path to the SQLite database containing the metadata.
        audio_dir:
            Path to the directory containing the audio files.
        additional_logging:
            Flag to turn on additional logging useful for debugging

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
        # "datetime_start",
        # "datetime_end",
        "text",
        "location",
        "location_roomdim",
        "noise_level",
        "noise_type",
        "source_url",
        "age",
        "gender",
        "dialect",
        # "language_native",
        # "language_spoken",
        "country_birth",
        # "zipcode_birth",
        # "zip_school",
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

    recording_ids: list[str] = [row[0] for row in rows]
    logger.info(f"Got {len(recording_ids)} recording ids")

    if num_metadata_samples != len(rows):
        logger.info(
            f"Expected to get all {num_metadata_samples} samples but got {len(rows)} "
            f"which means {num_metadata_samples - len(rows)} are missing"
        )
        if additional_logging:
            with sqlite3.connect(database=metadata_database_path) as conn:
                cursor = conn.execute("SELECT id_recording FROM Recordings")
                all_ids = [row[0] for row in list(map(list, cursor.fetchall()))]
                logger.info(
                    "The missing rows are "
                    f"{set(all_ids).difference(set(recording_ids))}"
                )

    # Get a list of all the audio file paths. We need this since the audio files lie in
    # subdirectories of the main audio directory
    audio_subdirs = list(audio_dir.iterdir())
    with Parallel(n_jobs=mp.cpu_count(), backend="threading") as parallel:
        all_audio_path_lists = parallel(
            delayed(list_files)(subdir)
            for subdir in tqdm(audio_subdirs, desc="Collecting audio file paths")
        )

    all_audio_paths_list = list(chain.from_iterable(all_audio_path_lists))
    all_audio_paths = {path.stem: path for path in all_audio_paths_list}
    logger.info(f"Got {len(all_audio_paths)} audio paths")

    # Match the audio files to the metadata, to ensure that there is a 1-to-1
    # correspondence between them
    logger.info("Matching the audio files to the metadata...")
    matched_audio_paths = [
        all_audio_paths.get(recording_id) for recording_id in recording_ids
    ]
    if len(recording_ids) != len(all_audio_paths):
        matching_audio_paths = [
            path for path in matched_audio_paths if path is not None
        ]
        ids_with_missing_paths = [
            id_ for id_, path in zip(recording_ids, matched_audio_paths) if path is None
        ]
        logger.info(f"Got {len(matching_audio_paths)} matched audio paths")
        logger.info(f"Got {len(ids_with_missing_paths)} missing audio paths")
        if additional_logging:
            logger.info(f"The missing paths are {ids_with_missing_paths}")
        if len(matching_audio_paths) != len(all_audio_paths):
            logger.info(
                f"Found {len(all_audio_paths)} audio paths but could only match "
                f"{len(matching_audio_paths)} of them to rows which means there are "
                f"{len(all_audio_paths) - len(matching_audio_paths)} too many audio "
                "paths"
            )
            if additional_logging:
                additional_paths = set(all_audio_paths.values()).difference(
                    set(matching_audio_paths)
                )
                logger.info(f"The additional paths are {additional_paths}")

    rows = [
        row + [str(audio_path)]
        for row, audio_path in zip(rows, matched_audio_paths)
        if audio_path is not None
    ]
    logger.info(f"Got {len(rows)} matched rows")

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


def list_files(
    audio_dir: Path, max_attempts: int = 10, extensions: list[str] = ["wav"]
) -> list[Path]:
    """List all files in the given directory, with a given set of extensions.

    Args:
        audio_dir:
            The directory containing the files.
        max_attempts (optional):
            The maximum number of attempts to list the files. Defaults to 10.
        extensions (optional):
            A list of extensions to consider when listing the files. Defaults to
            audio files with extension: ["wav"].

    Returns:
        A list of paths to the audio files.

    Raises:
        OSError:
            If the audio files cannot be listed.
    """
    for _ in range(max_attempts):
        try:
            return [file for ext in extensions for file in audio_dir.glob(f"*.{ext}")]
        except OSError:
            sleep(1)
    else:
        raise OSError(f"Failed to list the audio files in {audio_dir!r}.")


############################################
##### Building the conversation subset #####
############################################

# Constants
SPEAKER_A = "A"
SPEAKER_B = "B"
PATTERN_STARS = re.compile(r"\*\*\*(.*?)\*\*\*")
PATTERN_BRACKETS = re.compile(r"\[(.*?)\]")


def load_conv_speakers(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load speaker metadata from database that participate in conversations.

    Args:
        conn: SQLite database connection

    Returns:
        DataFrame with speaker information
    """
    query = """
        SELECT
            id_speaker,
            age,
            gender,
            dialect,
            country_birth,
            education,
            occupation
        FROM
            Speakers
        WHERE
            id_speaker IN (
                SELECT id_speaker_a FROM Conversations UNION
                SELECT id_speaker_b FROM Conversations UNION
                SELECT id_recorder FROM Conversations
            )
    """
    return pd.read_sql(query, conn)


def load_conversations_from_db(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load conversation metadata from database.

    Args:
        conn: SQLite database connection

    Returns:
        DataFrame with conversation information
    """
    query = """
        SELECT
            id_conversation,
            id_speaker_a,
            id_speaker_b,
            id_recorder,
            location,
            location_roomdim,
            noise_level,
            noise_type
        FROM
            Conversations
        WHERE
            id_speaker_a IN (SELECT id_speaker FROM Speakers) OR
            id_speaker_b IN (SELECT id_speaker FROM Speakers) OR
            id_recorder IN (SELECT id_speaker FROM Speakers)
    """
    return pd.read_sql(query, conn)


def build_file_path_dict(file_paths: List[Path]) -> Dict[str, Path]:
    """Build dictionary mapping file stems to paths.

    Args:
        file_paths: List of file paths

    Returns:
        Dictionary mapping stem to path
    """
    return {path.stem: path for path in file_paths}


def match_files_to_conversations(
    conversation_ids: pd.Series,
    file_paths_dict: Dict[str, Path],
    file_type: str,
    additional_logging: bool = False,
) -> Tuple[List[Path | None], Dict[str, int]]:
    """Match files to conversation IDs.

    Args:
        conversation_ids: Series of conversation IDs
        file_paths_dict: Dictionary mapping IDs to file paths
        file_type: Type of files being matched (for logging)
        additional_logging: Enable detailed logging

    Returns:
        Tuple of (matched paths list, statistics dict)
    """
    matched_paths: List[Path | None] = []
    missing_ids: List[Path | None] = []

    for conv_id in conversation_ids:
        path = file_paths_dict.get(conv_id)
        if path is not None:
            matched_paths.append(path)
        else:
            missing_ids.append(conv_id)
            matched_paths.append(None)

    stats = {
        "total_files": len(file_paths_dict),
        "matched": len([p for p in matched_paths if p is not None]),
        "missing": len(missing_ids),
        "extra": len(file_paths_dict)
        - len([p for p in matched_paths if p is not None]),
    }

    logger.info(f"Got {stats['matched']} matched {file_type} paths")
    logger.info(f"Got {stats['missing']} missing {file_type} paths")

    if additional_logging and missing_ids:
        logger.info(f"The missing {file_type} IDs are: {missing_ids}")

    if stats["extra"] > 0:
        logger.info(
            f"Found {stats['total_files']} {file_type} paths but could only match "
            f"{stats['matched']} of them, leaving {stats['extra']} unmatched"
        )
        if additional_logging:
            matched_paths_set = set(p for p in matched_paths if p)
            extra_paths = set(file_paths_dict.values()) - matched_paths_set
            logger.info(f"The additional {file_type} paths are: {extra_paths}")

    return matched_paths, stats


def get_speaker_info(df_speakers: pd.DataFrame, speaker_id: str) -> pd.Series:
    """Get speaker information by ID.

    Args:
        speaker_rows: DataFrame containing speaker data
        speaker_id: Speaker ID to look up

    Returns:
        Series with speaker information

    Raises:
        ValueError: If speaker not found or multiple matches
    """
    matches = df_speakers[df_speakers["id_speaker"] == str(speaker_id)]
    if len(matches) == 0:
        raise ValueError(f"Speaker {speaker_id} not found")
    if len(matches) > 1:
        raise ValueError(f"Multiple entries found for speaker {speaker_id}")
    return matches.iloc[0]


def is_valid_segment(text: str, speaker_label: str) -> bool:
    """Check if a transcript segment is valid for processing.

    Args:
        text: Transcript text
        speaker_label: Speaker identifier label from transcription (e.g., "A" or "B")

    Returns:
        True if segment should be processed, False otherwise
    """
    text = text.strip()

    # Skip empty
    if not text:
        return False

    # Skip text with special markers
    if PATTERN_STARS.search(text) or PATTERN_BRACKETS.search(text):
        return False

    # Validate speaker label
    speaker_label = speaker_label.strip().upper()
    if speaker_label not in (SPEAKER_A, SPEAKER_B):
        return False

    return True


def check_overlap(
    segment_index: int,
    segment_start: int,
    segment_end: int,
    transcription: pysubs2.SSAFile,
) -> bool:
    """Check if segment overlaps with adjacent segments.

    Args:
        segment_index: Index of current segment
        segment_start: Start time of segment
        segment_end: End time of segment
        transcription: Full transcription object

    Returns:
        True if overlap exists with previous or next segment
    """
    overlap_prev = False
    overlap_next = False

    # Check with previous segment
    if segment_index > 0:
        if segment_start < transcription[segment_index - 1].end:
            overlap_prev = True

    # Check with next segment
    if segment_index < len(transcription) - 1:
        if segment_end > transcription[segment_index + 1].start:
            overlap_next = True

    return overlap_prev or overlap_next


def extract_segment_data(
    segment: pysubs2.SSAEvent,
    segment_index: int,
    row_conversation: pd.Series,
    speaker_a: pd.Series,
    speaker_b: pd.Series,
    transcription: pysubs2.SSAFile,
    segment_path: Path,
) -> Dict:
    """Extract all data for a single segment.

    Args:
        segment: Transcription segment
        segment_index: Index of segment in conversation
        row_conversation: Row with conversation metadata
        speaker_a: Speaker A information
        speaker_b: Speaker B information
        transcription: Full transcription
        segment_path: Path where audio segment is saved

    Returns:
        Dictionary with segment data
    """
    speaker_label = segment.name.strip().upper()
    speaker_info = speaker_a if speaker_label == SPEAKER_A else speaker_b

    id_segment = f"{row_conversation.id_conversation}_{segment_index:05d}"

    has_overlap = check_overlap(
        segment_index, segment.start, segment.end, transcription
    )

    return {
        "id_conversation": id_segment,
        "location": row_conversation.location,
        "location_roomdim": row_conversation.location_roomdim,
        "noise_level": row_conversation.noise_level,
        "noise_type": row_conversation.noise_type,
        "id_speaker": speaker_info["id_speaker"],
        "age": speaker_info["age"],
        "gender": speaker_info["gender"],
        "dialect": speaker_info["dialect"],
        "country_birth": speaker_info["country_birth"],
        "education": speaker_info["education"],
        "occupation": speaker_info["occupation"],
        "overlap": has_overlap,
        "text": segment.text.strip(),
        "audio": str(segment_path),
    }


def process_single_conversation(
    row_conversation: pd.Series, df_speakers: pd.DataFrame, audio_dir: Path
) -> List[Dict]:
    """Process a single conversation and extract all segments.

    Args:
        row_conversation: Row with conversation metadata
        df_speakers: DataFrame with speaker information
        audio_dir: Base directory for audio files

    Returns:
        List of segment data dictionaries
    """
    try:
        speaker_a = get_speaker_info(df_speakers, row_conversation.id_speaker_a)
        speaker_b = get_speaker_info(df_speakers, row_conversation.id_speaker_b)
    except ValueError as e:
        logger.warning(f"Skipping conversation {row_conversation.id_conversation}: {e}")
        return []

    try:
        transcription = pysubs2.load(row_conversation.transcription_path)
        audio = AudioSegment.from_file(row_conversation.audio_path)
    except Exception as e:
        logger.error(
            f"Failed to load files for conversation "
            f"{row_conversation.id_conversation}: {e}"
        )
        return []

    # Create output directory
    dir_conversation = (
        audio_dir.parent
        / "conversation_segments"
        / Path(row_conversation.id_conversation)
    )
    dir_conversation.mkdir(parents=True, exist_ok=True)

    segment_entries = []

    for i, segment in enumerate(transcription):
        # Validate segment
        if not is_valid_segment(segment.text, segment.name):
            continue

        # Extract audio segment
        try:
            audio_clip = audio[segment.start : segment.end]
            segment_path = dir_conversation / f"{i}.wav"
            audio_clip.export(segment_path, format="wav")
        except Exception as e:
            logger.warning(
                f"Failed to export segment {i} in conversation "
                f"{row_conversation.id_conversation}: {e}"
            )
            continue

        # Extract segment data
        segment_data = extract_segment_data(
            segment,
            i,
            row_conversation,
            speaker_a,
            speaker_b,
            transcription,
            segment_path,
        )
        segment_entries.append(segment_data)

    return segment_entries


def process_all_conversations(
    df_conversations: pd.DataFrame, df_speakers: pd.DataFrame, audio_dir: Path
) -> pd.DataFrame:
    """Process all conversations and extract segments.

    Args:
        df_conversations: DataFrame with conversation metadata
        df_speakers: DataFrame with speaker information
        audio_dir: Base directory for audio files

    Returns:
        DataFrame with all processed segments
    """
    # debug limit
    df_conversations = df_conversations[:10]

    # Count total lines for progress bar
    total_lines = 0
    for row_conversation in df_conversations.itertuples():
        try:
            transcription = pysubs2.load(row_conversation.transcription_path)
            total_lines += len(transcription)
        except Exception as e:
            logger.warning(
                (
                    f"Failed to load transcription for "
                    f"{row_conversation.id_conversation}: {e}"
                )
            )

    logger.info(f"There are {total_lines:,} transcribed lines")

    all_segments = []

    with tqdm(
        total=total_lines, desc="Extracting audio segments", unit="segment"
    ) as pbar:
        for row_conversation in df_conversations.itertuples():
            segments = process_single_conversation(
                row_conversation, df_speakers, audio_dir
            )
            all_segments.extend(segments)

            # Update progress bar by number of lines in this conversation
            try:
                transcription = pysubs2.load(row_conversation.transcription_path)
                pbar.update(len(transcription))
            except Exception:
                pass

    logger.info(
        f"Got {len(all_segments):,} total transcribed segments which means "
        f"{total_lines - len(all_segments):,} got dropped"
    )

    return pd.DataFrame(all_segments)


def build_conversation_dataset(
    metadata_database_path: Path,
    audio_dir: Path,
    transcript_dir: Path,
    additional_logging: bool = False,
) -> Dataset:
    """Build the CoRal conversation dataset.

    Args:
        metadata_database_path:
            Path to the SQLite database containing the metadata.
        audio_dir:
            Path to the directory containing the audio files.
        transcript_dir:
            Path to the directory containing the transcription files.
        additional_logging:
            Flag to turn on additional logging useful for debugging

    Returns:
        The CoRal conversation dataset.
    """
    # Load metadata from database
    logger.info("Fetching the metadata from the SQLite database...")
    with sqlite3.connect(metadata_database_path) as conn:
        df_speakers = load_conv_speakers(conn)
        df_conversations = load_conversations_from_db(conn)

    logger.info(
        f"There are {len(df_conversations):,} conversations in the database with "
        f"{len(df_speakers)} distinct speakers"
    )

    # Collect and match audio files
    logger.info("Collecting audio file paths")
    audio_paths_list = list_files(audio_dir, extensions=["wav", "m4a"])
    audio_paths_dict = build_file_path_dict(audio_paths_list)
    logger.info(f"Got {len(audio_paths_dict)} audio paths")

    logger.info("Matching the audio files to the metadata...")
    matched_audio_paths, audio_stats = match_files_to_conversations(
        df_conversations["id_conversation"],
        audio_paths_dict,
        "audio",
        additional_logging,
    )

    df_conversations["audio_path"] = matched_audio_paths
    df_conversations = df_conversations.dropna(subset=["audio_path"])
    logger.info(f"Got {len(df_conversations)} total matched rows")

    # Collect and match transcription files
    logger.info("Collecting transcription file paths")
    transcript_paths_list = list_files(transcript_dir, extensions=["ass"])
    transcript_paths_dict = build_file_path_dict(transcript_paths_list)
    logger.info(f"Got {len(transcript_paths_dict)} transcription paths")

    logger.info("Matching the transcription files to the metadata...")
    matched_transcript_paths, transcript_stats = match_files_to_conversations(
        df_conversations["id_conversation"],
        transcript_paths_dict,
        "transcription",
        additional_logging,
    )

    df_conversations["transcription_path"] = matched_transcript_paths
    df_conversations = df_conversations.dropna(subset=["transcription_path"])
    logger.info(f"Got {len(df_conversations)} total matched rows")

    # Process all conversations and extract segments
    logger.info(
        "Extracting audio segments from each conversation based on transcriptions..."
    )
    processed_segments = process_all_conversations(
        df_conversations, df_speakers, audio_dir
    )

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(processed_segments)
    dataset = dataset.cast_column("audio", Audio())

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
    try:
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
    except ValueError as e:
        logger.error("Error during dataset splitting: %s", e)
        return None

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


################################################
##### Saving and uploading of the datasets #####
################################################


def save_dataset(
    read_aloud_dataset: DatasetDict | None,
    conversation_dataset: DatasetDict | None,
    dir_dest: Path | None = None,
) -> None:
    """Save the datasets to local disk.

    Args:
        read_aloud_dataset:
            The read-aloud dataset, or None if no such dataset exists.
        conversation_dataset:
            The conversation dataset, or None if no such dataset exists.
        dir_dest:
            The directory where the datasets should be saved.
    """
    if dir_dest is None:
        dir_dest = Path.cwd() / "data" / "processed"
    dir_dest.mkdir(parents=True, exist_ok=True)

    if read_aloud_dataset is not None:
        read_aloud_dataset.save_to_disk(str(dir_dest / "coral_read_aloud_dataset"))

    if conversation_dataset is not None:
        conversation_dataset.save_to_disk(str(dir_dest / "coral_conversation_dataset"))


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

    Returns:
        The new directory containing the audio files.
    """
    new_audio_dir = Path.cwd() / "data" / "raw" / audio_dir.name
    new_audio_dir.mkdir(exist_ok=True)

    # Get list of subdirectories of the audio directory, or abort of none exist
    audio_subdirs = [path for path in audio_dir.iterdir() if path.is_dir()]
    if not audio_subdirs:
        return new_audio_dir

    # Compress all subdirectories
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


def copy_files_to_cwd(
    source_dir: Path, use_compression: bool = False, dest_dir: Path | None = None
) -> Path:
    """Copies files from the source directory to cwd data directory.

    Args:
        source_dir (Path): The source directory from which files will be copied.
        use_compression (bool, optional): Compresses the files before copying.
        dest_dir (Path, optional): The destination directory where files will be copied.

    Returns:
        Path (Path): The destination directory where files have been copied.
    """
    if dest_dir is None:
        dest_dir = Path.cwd() / "data" / "raw" / source_dir.name

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Identify files to be copied
    files_to_copy = [
        file
        for file in source_dir.rglob("*")
        if file.is_file() and file.suffix != ".tar.xz"
    ]

    if use_compression:
        tar_path = source_dir / "temp_transfer.tar.xz"

        compress_files(files_to_copy, tar_path)
        decompress_file(tar_path, dest_dir)

    else:

        def copy_func(file: Path) -> None:
            """Copy a file to the destination directory."""
            target = dest_dir / file.name

            if target.exists():
                # Quick check: size & mtime
                src_stat = file.stat()
                dst_stat = target.stat()
                if (
                    src_stat.st_size == dst_stat.st_size
                    and src_stat.st_mtime == dst_stat.st_mtime
                ):
                    return  # Skip identical file

            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)

        # Copy files without compression, flattening structure
        with Parallel(n_jobs=mp.cpu_count(), backend="threading") as parallel:
            parallel(
                delayed(copy_func)(file)
                for file in tqdm(files_to_copy, desc="Copying files")
            )

    return dest_dir


def compress_files(file_paths: list[Path], path_destination: Path) -> Path:
    """Compress files from a list of paths using tarfile, to a new directory.

    Args:
        file_paths:
            A list of paths to the files to compress.
        path_destination:
            The directory to save the compressed file.
    """
    path_destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path_destination, mode="w:xz") as tar:
        for file_path in file_paths:
            tar.add(file_path, arcname=file_path.name)
    return path_destination


def compress_dir(directory: Path) -> Path:
    """Compress a directory using tar.

    Args:
        directory:
            The directory to compress.

    Returns:
        The path to the compressed file.
    """
    # if not directory.with_suffix(".tar.xz").exists():
    archive_dir = f"{str(directory)}.tar.xz"
    with tarfile.open(name=archive_dir, mode="w:xz") as tar:
        tar.add(name=directory, arcname=directory.name)

    return Path(archive_dir)


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
