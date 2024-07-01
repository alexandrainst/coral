"""Script that builds and uploads the CoRal speech recognition dataset from the raw data.

Usage:
    python src/scripts/build_coral_asr.py --nas-dir <path/to/nas/dir> --hub-id <hub-id>
"""

import logging

import click

logger = logging.getLogger("build_coral_asr")


@click.command()
@click.option(
    "--nas-dir",
    type=click.Path(exists=True),
    default="/Volumes/CoRal",
    show_default=True,
    help="Path to the Mounted Network Attached Storage (NAS) containing CoRal data.",
)
@click.option(
    "--audio-dir",
    type=click.Path(exists=True),
    default="_new_structure/raw/recordings",
    show_default=True,
    help="Relative path to the directory containing the audio files.",
)
@click.option(
    "--metadata-database-path",
    type=click.Path(exists=True),
    default="_new_structure/raw/CoRal_public.db",
    show_default=True,
    help="Path to the SQLite database containing the metadata.",
)
@click.option(
    "--hub-id",
    type=str,
    default="alexandrainst/coral",
    show_default=True,
    help="Identifier of the Hugging Face Hub repository.",
)
def main(
    nas_dir: str, audio_dir: str, metadata_database_path: str, hub_id: str
) -> None:
    """Build and upload the CoRal speech recognition dataset."""
    pass


if __name__ == "__main__":
    main()
