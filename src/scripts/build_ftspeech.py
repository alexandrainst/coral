"""Script that builds the FTSpeech dataset from the raw data.

Usage:
    python build_ftspeech.py <raw_data_dir> <output_dir>
"""

import click

from coral_models.asr.ftspeech import build_and_store_ftspeech


@click.command("Builds and stores the FTSpeech dataset.")
@click.argument(
    "raw_data_dir",
    type=click.Path(exists=True),
)
@click.argument(
    "output_dir",
    type=click.Path(),
)
def main(raw_data_dir: str, output_dir: str) -> None:
    """Builds and stores the FTSpeech dataset."""
    build_and_store_ftspeech(input_dir=raw_data_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
