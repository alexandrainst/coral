"""Checks if a list of audio clips can be opened, and prints the list of bad ones.

Usage:
    python src/scripts/find_faulty_audio_clips.py <path/to/audio/dir>
"""

import logging
import warnings
from pathlib import Path

import click
import librosa
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.argument("audio_dir", type=click.Path(exists=True))
def main(audio_dir: str | Path) -> None:
    """Checks if a list of audio clips can be opened and prints the list of bad ones.

    Args:
        audio_dir: Path to directory containing audio clips.
    """
    audio_dir = Path(audio_dir)
    folders = [folder for folder in audio_dir.iterdir() if folder.is_dir()]
    folders += [audio_dir]
    for folder in tqdm(folders, desc=f"Checking {audio_dir} for faulty audio files"):
        for audio_file in tqdm(
            iterable=list(folder.glob("*.wav")),
            desc=f"Checking {folder}",
            leave=False,
        ):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    warnings.simplefilter("ignore", category=FutureWarning)
                    librosa.load(audio_file)
            except Exception as e:
                logger.error(f"Could not open {audio_file!r}. The error was: {e}")


if __name__ == "__main__":
    main()
