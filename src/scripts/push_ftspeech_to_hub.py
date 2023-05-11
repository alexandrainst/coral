"""Script that pushes a compiled FTSpeech dataset to the Hugging Face Hub.

Usage:
    python push_ftspeech_to_hub.py <compiled_dataset_dir>
"""

import click
from datasets import DatasetDict


@click.command(help="Pushes a compiled FTSpeech dataset to the Hugging Face Hub.")
@click.argument(
    "compiled-dataset-dir",
    type=click.Path(exists=True),
    help="The directory where the compiled dataset is stored.",
)
@click.option(
    "--dataset-id",
    type=str,
    default="alexandrainst/ftspeech",
    show_default=True,
    help="The ID of the dataset on the Hugging Face Hub.",
)
@click.option(
    "--private",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to make the dataset private on the Hugging Face Hub.",
)
def main(compiled_data_dir: str, dataset_id: str, private: bool) -> None:
    """Builds and stores the FTSpeech dataset.

    This also catches RuntimeError exceptions which tends to happen during the upload
    of large datasets, and retries the upload until it succeeds.

    Args:
        compiled_data_dir (str):
            The directory where the compiled dataset is stored.
        dataset_id (str):
            The ID of the dataset on the Hugging Face Hub.
        private (bool):
            Whether to make the dataset private.
    """
    dataset = DatasetDict.load_from_disk(compiled_data_dir)
    while True:
        try:
            dataset.push_to_hub(
                repo_id=dataset_id,
                max_shard_size="50MB",
                private=private,
            )
            break
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()
