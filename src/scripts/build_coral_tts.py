"""Script that builds the CoRal text-to-speech dataset from the raw data.

Usage:
    python src/scripts/build_coral_tts.py TRANSCRIPTION_TXT_FILE AUDIO_DIR OUTPUT_DIR
"""

import logging
import multiprocessing as mp
from pathlib import Path

import click
import pandas as pd
from datasets import Audio, Dataset, DatasetDict
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_coral_tts")


SAMPLE_RATE = 44_100


@click.command()
@click.argument("transcription_file", type=click.Path(exists=True))
@click.argument("audio_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def main(
    transcription_file: str | Path, audio_dir: str | Path, output_dir: str | Path
) -> None:
    """Builds and stores the CoRal text-to-speech dataset.

    Args:
        transcription_file:
            Path to the transcription file.
        audio_dir:
            Path to the directory containing the audio files.
        output_dir:
            Path to the directory where the dataset will be stored.
    """
    transcription_file = Path(transcription_file)
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)

    with transcription_file.open() as f:
        transcriptions = [line.strip() for line in f.readlines()]

    records = list()
    audio_paths = sorted(audio_dir.glob("*.wav"), key=lambda x: x.stem.split("_"))
    for path in tqdm(audio_paths, desc="Processing audio files"):
        speaker_id = path.stem.split("_")[0]
        transcription_id = int(path.stem.split("_")[1]) - 1
        if transcription_id < 0:
            continue

        # Fix an error in the indexing of the one of the speakers
        if speaker_id == "mic" and transcription_id >= 11100:
            transcription_id -= 101
        if speaker_id == "mic" and transcription_id >= 11200:
            transcription_id += 1

        transcription = transcriptions[transcription_id].strip()
        record = dict(
            speaker_id=speaker_id,
            transcription_id=transcription_id,
            text=transcription,
            audio=str(path.resolve()),
        )
        records.append(record)

    df = pd.DataFrame.from_records(records)
    dataset = Dataset.from_pandas(df, preserve_index=False).cast_column(
        "audio", Audio(sampling_rate=SAMPLE_RATE)
    )

    dataset_dict = DatasetDict(dict(train=dataset))
    dataset_dict.save_to_disk(
        str(output_dir), max_shard_size="500MB", num_proc=mp.cpu_count() - 1
    )


if __name__ == "__main__":
    main()
