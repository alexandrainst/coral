"""Script that pushes a saved dataset to the Hugging Face Hub.

Usage:
    python src/scripts/push_to_hub.py SAVED_DATASET_DIR HUB_ID [--private]
"""

import logging
import time

import click
from datasets import DatasetDict
from requests import HTTPError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("push_to_hub")


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

    Args:
        saved_data_dir:
            The directory where the saved dataset is stored.
        hub_id:
            The Hugging Face Hub ID to push the dataset to.
        private:
            Whether to make the dataset private on the Hugging Face Hub.
    """
    dataset = DatasetDict.load_from_disk(saved_data_dir)
    while True:
        try:
            dataset.push_to_hub(repo_id=hub_id, max_shard_size="500MB", private=private)
            break
        except (RuntimeError, HTTPError) as e:
            logger.error(f"Error while pushing to hub: {e}")
            logger.info("Waiting a minute before trying again...")
            time.sleep(60)
            logger.info("Retrying...")


if __name__ == "__main__":
    main()
