"""Functions related to building the FTSpeech dataset."""

import multiprocessing as mp
from pathlib import Path

import pandas as pd
from datasets import Audio, Dataset, DatasetDict
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pydub import AudioSegment
from tqdm.auto import tqdm


def preprocess_transcription(transcription: str) -> str:
    """Preprocess a transcription.

    Args:
        transcription (str):
            The transcription to preprocess.

    Returns:
        str:
            The preprocessed transcription.
    """
    # Strip the transcription of <UNK> tokens
    transcription = transcription.replace("<UNK>", "")

    # Remove trailing whitespace
    transcription = transcription.strip()

    # Return the preprocessed transcription
    return transcription


def split_audio(record: dict, input_path: str | Path) -> None:
    """Loads a full audio clip and splits it according to the record.

    Args:
        record (dict):
            The record containing the audio data. Needs to have columns
            `utterance_id`, `start_time` and `end_time`.
        input_path (str or Path):
            The path to the directory where the raw dataset is stored.
    """
    # Ensure that `input_path` is a Path object
    input_path = Path(input_path)

    # Build the audio file path
    year: str = record["utterance_id"].split("_")[1][:4]
    filename = "_".join(record["utterance_id"].split("_")[1:3]) + ".wav"
    audio_path: Path = input_path / "audio" / year / filename

    # Get the start and end times in milliseconds
    start_time: int = record["start_time"] * 1000
    end_time: int = record["end_time"] * 1000

    # Load and split the audio
    audio: AudioSegment = AudioSegment.from_wav(str(audio_path))[start_time:end_time]

    # Store the audio
    new_filename: str = record["utterance_id"] + ".wav"
    new_audio_path: Path = input_path / "processed_audio" / new_filename
    out_ = audio.export(str(new_audio_path.resolve()), format="wav")
    out_.close()

    # Close audio
    del audio


def build_and_store_data(
    cfg: DictConfig, input_dir: Path | str, output_dir: Path | str
) -> None:
    """Builds and saves the FTSpeech dataset.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
        input_dir (str or Path):
            The directory where the raw dataset is stored.
        output_dir (str or Path):
            The path to the resulting dataset.

    Raises:
        FileNotFoundError:
            If `input_dir` does not exist.
    """
    # Ensure that the paths are Path objects
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # If `input_dir` does not exist, raise an error
    if not input_dir.exists():
        raise FileNotFoundError(f"{input_dir} does not exist")

    # Set up paths to the transcription data
    paths = {
        "train": input_dir / "text" / "ft-speech_train.tsv",
        "dev_balanced": input_dir / "text" / "ft-speech_dev-balanced.tsv",
        "dev_other": input_dir / "text" / "ft-speech_dev-other.tsv",
        "test_balanced": input_dir / "text" / "ft-speech_test-balanced.tsv",
        "test_other": input_dir / "text" / "ft-speech_test-other.tsv",
    }

    # Load file with transcriptions
    dfs = {split: pd.read_csv(path, sep="\t") for split, path in paths.items()}

    # Preprocess the transcriptions
    for split, df in dfs.items():
        df[cfg.dataset.text_column] = df.transcript.map(preprocess_transcription)
        dfs[split] = df

    # Add a `speaker_id` column to the dataframes
    for split, df in dfs.items():
        df["speaker_id"] = [row.utterance_id.split("_")[0] for _, row in df.iterrows()]
        dfs[split] = df

    # Ensure that `processed_audio` exists
    processed_audio_path = input_dir / "processed_audio"
    if not processed_audio_path.exists():
        processed_audio_path.mkdir()

    # Split the audio files
    for split, df in tqdm(list(dfs.items()), desc="Splitting audio"):
        pbar = tqdm(df.to_dict("records"), desc=split, leave=False)
        split_fn = delayed(split_audio)
        with Parallel(n_jobs=mp.cpu_count()) as parallel:
            parallel(split_fn(record, input_dir) for record in pbar)

    # Add an `audio` column to the dataframes, containing the paths to the
    # audio files
    for split, df in dfs.items():
        audio_paths = list()
        for _, row in df.iterrows():
            filename = row.utterance_id + ".wav"
            audio_path = input_dir / "processed_audio" / filename
            audio_paths.append(str(audio_path.resolve()))
        df["audio"] = audio_paths
        dfs[split] = df

    # Remove unused columns
    cols_to_drop = ["utterance_id", "start_time", "end_time", "transcript"]
    for split, df in dfs.items():
        df = df.drop(columns=cols_to_drop)
        dfs[split] = df

    # Convert the dataframe to a HuggingFace Dataset
    datasets = {
        split: Dataset.from_pandas(df, preserve_index=False)
        for split, df in dfs.items()
    }
    dataset = DatasetDict(datasets)

    # Cast `audio` as the audio path column
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    # Store the dataset
    dataset.save_to_disk(str(output_dir))
