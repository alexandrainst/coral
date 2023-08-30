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
    folders = [folder for folder in audio_dir.iterdir() if folder.is_dir()]
    folders += [audio_dir]
    for folder in folders:
        for audio_file in tqdm(list(folder.glob("*.wav")), desc=f"Checking {folder}"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    warnings.simplefilter("ignore", category=FutureWarning)
                    librosa.load(audio_file)
            except Exception as e:
                print(audio_file, e)


if __name__ == "__main__":
    main()
