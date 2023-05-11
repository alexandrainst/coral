"""Script that pushes a compiled FTSpeech dataset to the Hugging Face Hub.

Usage:
    python push_ftspeech_to_hub.py <compiled_dataset_dir>
"""

import click
from datasets import DatasetDict


@click.command()
@click.argument("compiled_dataset_dir", type=click.Path(exists=True))
@click.option("--dataset-id", type=str, default="alexandrainst/ftspeech")
def main(compiled_data_dir: str, dataset_id: str) -> None:
    """Builds and stores the FTSpeech dataset.

    Args:
        compiled_data_dir (str):
            The directory where the compiled dataset is stored.
        dataset_id (str):
            The ID of the dataset on the Hugging Face Hub.
    """
    dataset = DatasetDict.load_from_disk(compiled_data_dir)
    while True:
        try:
            dataset.push_to_hub(repo_id=dataset_id, max_shard_size="50MB")
            break
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()
