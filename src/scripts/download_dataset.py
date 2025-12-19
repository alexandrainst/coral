"""Download a HF dataset to disk."""

from pathlib import Path

import click
from datasets import load_dataset


@click.command()
@click.argument("dataset_id")
@click.argument("output_dir")
def download_dataset(dataset_id: str, output_dir: str) -> None:
    """Download a dataset from the Hugging Face Hub to disk.

    Args:
        dataset_id:
            The identifier of the dataset to download.
        output_dir:
            The directory where the dataset will be saved.
    """
    ds = load_dataset(dataset_id)
    ds.save_to_disk(Path(output_dir, dataset_id.replace("/", "--")))


if __name__ == "__main__":
    download_dataset()
