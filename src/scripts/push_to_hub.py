"""Script that pushes a saved dataset to the Hugging Face Hub.

Usage:
    python push_to_hub.py <saved_dataset_dir> <hub_id> [--private]
"""

import click
from datasets import DatasetDict


@click.command("Pushes a saved dataset to the Hugging Face Hub.")
@click.argument("saved-data-dir", type=click.Path(exists=True))
@click.argument("hub-id", type=str)
@click.option(
    "--private",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to make the dataset private on the Hugging Face Hub.",
)
def main(saved_data_dir: str, hub_id: str, private: bool) -> None:
    """Pushes a saved dataset to the Hugging Face Hub.

    This also catches RuntimeError exceptions which tends to happen during the upload
    of large datasets, and retries the upload until it succeeds.
    """
    dataset = DatasetDict.load_from_disk(saved_data_dir)
    while True:
        try:
            dataset.push_to_hub(
                repo_id=hub_id,
                max_shard_size="50MB",
                private=private,
            )
            break
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()
