"""Script that builds a synthetic vocie audio from reading 
the nst dataset. 

Usage:
    python src/scripts/build_synthetic_nts.py --method gtts ./data/raw_data/nst-da-train-metadata.csv ./data/raw_data/
"""

from pathlib import Path
import subprocess

from gtts import gTTS
import pandas as pd
import click


def generate_speech_mac(text: str, filename: Path):
    """Generate speech from text using macOS 'say' command and save it to a file.

    Note, these voice has a licens only for private use.
    The downloaded voices needs to be changed manually in
    spoken content in settings of the mac machine.
    The file is saved in a special mac sound format called ".aiff".
    Extra details see:
    "https://maithegeek.medium.com/having-fun-in-macos-with-say-command-d4a0d3319668"

    Args:
        text: The text to convert to speech.
        filename: The name of the output audio file.

    Returns:
        None
    """
    subprocess.run(["say", text, "-o", filename])


def generate_speech_espeak(text: str, filename: Path, variant: str = "+m1"):
    """Generate speech from text using eSpeak and save it to a file.

    Args:
        text: The text to convert to speech.
        filename: The name of the output audio file.
        variant: The eSpeak voice variant. Default is "+m1".

    Returns:
        None
    """
    subprocess.run(["espeak", "-vda", "-w", filename, variant, text])


def generate_speech_gtts(text: str, filename: Path, language: str = "da"):
    """Generate speech from text using gTTS and save it to a file.

    Args:
        text: The text to convert to speech.
        filename: The name of the output audio file.
        language (str, optional): Language used to speek, default is 'da' (Danish).

    Returns:
        None
    """
    tts = gTTS(text, lang=language)
    tts.save(filename)


@click.command()
@click.option('--method', type=click.Choice(['mac', 'espeak', 'gtts']), default='gtts',
              help='Choose the method for generating speech')
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(method, input_file: Path, output_dir: Path):
    """Script that builds a synthetic voice audio from reading the nst dataset."""
    # Read the Excel file into a pandas DataFrame
    columns = ["audio", "text", "speaker_id", "age",
               "sex", "dialect", "recording_datetime"]
    df = pd.read_csv(input_file, usecols=columns)

    for index, row in df.iterrows():
        if pd.isna(row["text"]):
            pass
        else:
            text_danish = row["text"]
            filename = Path(output_dir) / row["audio"]
            if method == 'mac':
                generate_speech_mac(text_danish, filename)
            elif method == 'espeak':
                generate_speech_espeak(text_danish, filename)
            elif method == 'gtts':
                generate_speech_gtts(text_danish, filename)
            else:
                pass


if __name__ == "__main__":
    main()
