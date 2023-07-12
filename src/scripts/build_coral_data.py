"""Script that preprocesses the raw CoRal data.

Usage:
    python build_coral_data.py <input_path> <metadata_path> <output_path>
"""
import click

from coral_models.prepare_raw_data import prepare_raw_data


@click.command("Preprocesses the CoRal dataset.")
@click.argument(
    "input_path",
    type=click.Path(exists=True),
)
@click.option(
    "metadata_path",
    type=click.Path(exists=True),
)
@click.argument(
    "output_path",
    type=click.Path(),
)
def main(input_path: str, output_path: str, metadata_path: str) -> None:
    prepare_raw_data(input_path, output_path, metadata_path)


if __name__ == "__main__":
    main()
