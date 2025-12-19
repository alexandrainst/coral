"""Download a HF dataset to disk.

Usage:
    uv run download_dataset.py --dataset-id DATASET_ID --output-dir OUTPUT_DIR
"""

from pathlib import Path
from shutil import rmtree

import click
from huggingface_hub._snapshot_download import snapshot_download


@click.command()
@click.option(
    "--dataset-id",
    type=str,
    required=True,
    help="The identifier of the dataset to download from the Hugging Face Hub.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The directory where the dataset directory will be saved.",
)
def download_dataset(dataset_id: str, output_dir: Path) -> None:
    """Download a dataset from the Hugging Face Hub to disk.

    Args:
        dataset_id:
            The identifier of the dataset to download.
        output_dir:
            The directory where the dataset will be saved.
    """
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=output_dir / dataset_id.replace("/", "--"),
        cache_dir=output_dir / ".cache",
    )
    rmtree(output_dir / ".cache")


if __name__ == "__main__":
    download_dataset()
