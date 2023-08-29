"""Checks if a list of audio clips can be opened, and prints the list of bad ones.

Usage:
    python src/scripts/find_faulty_audio_clips.py <path/to/audio/dir>
"""

import warnings
from pathlib import Path

import click
import librosa
from tqdm.auto import tqdm


@click.command()
@click.argument("audio_dir", type=click.Path(exists=True))
def main(audio_dir: str | Path) -> None:
    """Checks if a list of audio clips can be opened and prints the list of bad ones.

    Args:
        audio_dir: Path to directory containing audio clips.
    """
    audio_dir = Path(audio_dir)
    for audio_file in tqdm(list(audio_dir.glob("*.wav")), desc=f"Checking {audio_dir}"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                librosa.load(audio_file)
        except Exception as e:
            print(audio_file, e)


if __name__ == "__main__":
    main()
