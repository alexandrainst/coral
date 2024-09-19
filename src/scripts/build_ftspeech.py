"""Script that builds the FTSpeech dataset from the raw data.

Usage:
    python src/scripts/build_ftspeech.py RAW_DATA_DIR OUTPUT_DIR
"""

import logging
import multiprocessing as mp
from pathlib import Path

import click
import pandas as pd
from datasets import Audio, Dataset, DatasetDict
from pydub import AudioSegment
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_ftspeech")


@click.command("Builds and stores the FTSpeech dataset.")
@click.argument("raw_data_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def main(raw_data_dir: str | Path, output_dir: str | Path) -> None:
    """Builds and stores the FTSpeech dataset.

    Args:
        raw_data_dir:
            The directory where the raw dataset is stored.
        output_dir:
            The path to the resulting dataset.

    Raises:
        FileNotFoundError:
            If `input_dir` does not exist.
    """
    input_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"{input_dir} does not exist")

    # These are the paths to the transcription files
    paths = {
        "train": input_dir / "text" / "ft-speech_train.tsv",
        "dev_balanced": input_dir / "text" / "ft-speech_dev-balanced.tsv",
        "dev_other": input_dir / "text" / "ft-speech_dev-other.tsv",
        "test_balanced": input_dir / "text" / "ft-speech_test-balanced.tsv",
        "test_other": input_dir / "text" / "ft-speech_test-other.tsv",
    }

    logger.info("Loading transcription files...")
    dfs = {split: pd.read_csv(path, sep="\t") for split, path in paths.items()}

    logger.info("Preprocessing the transcription files...")
    for split, df in dfs.items():
        df["sentence"] = df.transcript.map(preprocess_transcription)
        dfs[split] = df

    # Add a `speaker_id` column to the dataframes
    for split, df in dfs.items():
        df["speaker_id"] = [row.utterance_id.split("_")[0] for _, row in df.iterrows()]
        dfs[split] = df

    # Ensure that `processed_audio` exists
    processed_audio_path = input_dir / "processed_audio"
    processed_audio_path.mkdir(exist_ok=True)

    # Split the audio files
    for split, df in tqdm(list(dfs.items()), desc="Splitting audio"):
        df["src_fname"] = df.utterance_id.map(
            lambda id_str: "_".join(id_str.split("_")[1:3])
        )
        for src_name in tqdm(df.src_fname.unique(), desc=split, leave=False):
            split_audio(
                records=df.query("src_fname == @src_name").to_dict("records"),
                input_dir=input_dir,
            )

    # Add an `audio` column to the dataframes, containing the paths to the audio files
    for split, df in dfs.items():
        audio_paths: list[str] = list()
        for _, row in df.iterrows():
            filename: str = row.utterance_id + ".wav"
            audio_path: Path = input_dir / "processed_audio" / filename
            audio_paths.append(str(audio_path.resolve()))
        df["audio"] = audio_paths
        dfs[split] = df

    # Remove unused columns
    cols_to_drop = ["src_fname", "start_time", "end_time", "transcript"]
    for split, df in dfs.items():
        df = df.drop(columns=cols_to_drop)
        dfs[split] = df

    # Convert the dataframe to a HuggingFace Dataset
    datasets = {
        split: Dataset.from_pandas(df, preserve_index=False)
        for split, df in dfs.items()
    }
    dataset = DatasetDict(datasets)

    # Cast `audio` as the audio path column, which ensures that the audio can be loaded
    # seamlessly on the fly
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    logger.info(f"Saving the dataset to {output_dir}...")
    dataset.save_to_disk(
        str(output_dir), max_shard_size="500MB", num_proc=mp.cpu_count() - 1
    )


def preprocess_transcription(transcription: str) -> str:
    """Preprocess a transcription.

    Args:
        transcription (str):
            The transcription to preprocess.

    Returns:
        str:
            The preprocessed transcription.
    """
    # Strip the transcription of <UNK> tokens, since these are not useful for training
    # a speech recognition model
    transcription = transcription.replace("<UNK>", "")

    # Remove trailing whitespace and newlines
    transcription = transcription.strip()

    return transcription


def split_audio(records: list[dict], input_dir: str | Path) -> None:
    """Loads a full audio clip and splits it according to the record.

    Args:
        records (list[dict]):
            A list of records, each containing the following keys: `utterance_id`,
            `start_time`, `end_time`. It is assumed that they all refer to the same
            audio clip.
        input_dir (str or Path):
            The path to the directory where the raw dataset is stored.
    """
    # Ensure that `input_dir` is a Path object
    input_dir = Path(input_dir)

    # Build the path where we will save the audio
    processed_audio_dir = input_dir / "processed_audio"
    new_audio_paths: list[Path] = [
        processed_audio_dir / (record["utterance_id"] + ".wav") for record in records
    ]

    # If the audio file already exists, we don't need to do anything
    if all(new_audio_path.exists() for new_audio_path in new_audio_paths):
        return

    # Load the audio
    _, year_with_a_one_at_the_end, code, _ = records[0]["utterance_id"].split("_")
    year: str = year_with_a_one_at_the_end[:4]
    filename = f"{year_with_a_one_at_the_end}_{code}.wav"
    audio_path: Path = input_dir / "audio" / year / filename
    audio = AudioSegment.from_wav(str(audio_path))
    assert isinstance(audio, AudioSegment)

    for record in records:
        split_single_audio(
            audio=audio, record=record, processed_audio_dir=processed_audio_dir
        )


def split_single_audio(
    processed_audio_dir: Path, record: dict, audio: AudioSegment
) -> None:
    """Split an audio file as according to a record containing start and end times.

    Args:
        processed_audio_dir (Path):
            The data directory where the processed audio files are stored.
        record (dict):
            The record, which has to have the keys "utterance_id", "start_time" and
            "end_time".
        audio (AudioSegment):
            The audio to split.
    """
    new_audio_path = processed_audio_dir / (record["utterance_id"] + ".wav")

    # Get the start and end times in milliseconds, as `pydub` works with audio
    # files in milliseconds
    start_time: int = record["start_time"] * 1000
    end_time: int = record["end_time"] * 1000

    # Load the audio and slice it according to the start and end times in the
    # record
    audio_segment = audio[start_time:end_time]
    assert isinstance(audio_segment, AudioSegment)

    # Store the sliced audio
    out_ = audio_segment.export(str(new_audio_path.resolve()), format="wav")
    out_.close()
    del audio_segment


if __name__ == "__main__":
    main()
