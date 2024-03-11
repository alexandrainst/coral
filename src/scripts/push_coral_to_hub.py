"""Pushes the processed CoRal dataset to the Hugging Face Hub.

Usage:
    python src/scripts/push_coral_to_hub.py\
            <path/to/recording/metadata/path> <path/to/speaker/metadata/path> <hub_id>
"""

from pathlib import Path
from datasets import Dataset, Audio, DatasetDict
from datetime import datetime
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepositoryNotFoundError
import pandas as pd
import click
import logging
import time

from requests import HTTPError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

FIRST_ITERATION_END = "2024-03-15T00:00:00+02:00"
TEST_SPEAKER_IDS = [
    "t16023910",
    "t22996947",
    "t17053799",
    "t47310812",
    "t59821643",
    "t79384144",
    "t37263163",
    "t39126337",
    "t22996947",
    "t17419826",
    "t29982093",
    "t40224720",
    "t27567090",
    "t31776488",
    "t40353825",
    "t34653502",
    "t26322580",
    "t72771561",
    "t48392154",
    "t38841910",
    "t36170006",
    "t10062436",
    "t21820268",
    "t39414039",
    "t13282221",
    "t82923090",
    "t35107011",
    "t13330719",
    "t33840200",
    "t10367179",
    "t39656524",
    "t37619007",
    "t42345465",
    "t21902156",
]


@click.command("Builds and pushes the CoRal test dataset.")
@click.option(
    "--recording_metadata_path",
    type=click.Path(exists=True),
    default="data/processed/recordings.xlsx",
    show_default=True,
    help="The path to the recording metadata.",
)
@click.option(
    "--speaker_metadata_path",
    type=click.Path(exists=True),
    default="data/hidden/speakers.xlsx",
    show_default=True,
    help="The path to the speaker metadata.",
)
@click.option(
    "--hub_id",
    type=str,
    default="alexandrainst/coral",
    show_default=True,
    help=(
        "The Hugging Face Hub id. Note that the version number will be appended to"
        " this id.",
    ),
)
@click.option(
    "--major_version",
    type=int,
    show_default=True,
    help="The major version number of the dataset.",
)
@click.option(
    "--minor_version",
    type=int,
    show_default=True,
    help="The minor version number of the dataset.",
)
@click.option(
    "--private",
    is_flag=True,
    default=True,
    show_default=True,
    help="Whether to make the dataset private on the Hugging Face Hub.",
)
def main(
    recording_metadata_path: str | Path,
    speaker_metadata_path: str | Path,
    hub_id: str,
    major_version: int,
    minor_version: int,
    private: bool,
) -> None:

    # Load the metadata and split into test/train speakers
    recording_metadata_path = Path(recording_metadata_path)
    recording_metadata = pd.read_excel(recording_metadata_path, index_col=0)

    # Change the dtype of all columns to string
    recording_metadata = recording_metadata.astype(str)

    # Define dictionary that defines CoRal iteration boundaries
    iteration_periods: dict[str, tuple[datetime, datetime]] = {
        "iteration_1": (datetime.min, timestamp(FIRST_ITERATION_END)),
        "iteration_2": (timestamp(FIRST_ITERATION_END), datetime.max),
    }

    def get_iteration_name(start_time: str | datetime) -> str:
        """Returns the CoRal iteration name of a recording start time.

        Args:
            start_time:
                The starting time of the recording.

        Returns:
            The iteration name.

        Raises:
            ValueError:
                If no iteration name could be associated to the starting time.
        """
        if isinstance(start_time, str):
            start_time = timestamp(start_time)

        for iteration_name, (
            iteration_start,
            iteration_end,
        ) in iteration_periods.items():
            if start_time >= iteration_start and start_time < iteration_end:
                return iteration_name
        else:
            raise ValueError(
                f"The start time {start_time} doesn't correspond to any iteration!"
            )

    recording_metadata["iteration"] = recording_metadata["start"].apply(
        get_iteration_name
    )

    test_recordings = []
    train_recordings = []
    for _, row in recording_metadata.iterrows():
        if any(
            [
                speaker_id in TEST_SPEAKER_IDS
                for speaker_id in row["speaker_id"].split(",")
            ]
        ):
            test_recordings.append(row)
        else:
            train_recordings.append(row)
    test_recordings_df = pd.DataFrame(test_recordings)
    train_recordings_df = pd.DataFrame(train_recordings)

    # Load the speaker metadata
    speaker_metadata_path = Path(speaker_metadata_path)
    speaker_metadata = pd.read_excel(speaker_metadata_path, index_col=0)

    # Change the dtype of all columns to string
    speaker_metadata = speaker_metadata.astype(str)

    # Remove columns from the speaker metadata, and create age groups
    speaker_metadata.drop(
        columns=[
            "name",
            "mail",
            "zip_primary_school",
            "zip_grew_up",
            "birthplace",
            "education",
            "occupation",
        ],
        inplace=True,
    )
    speaker_metadata["age"] = (
        speaker_metadata["age"]
        .astype(int)
        .apply(lambda x: "<25" if x < 25 else "25-50" if x < 50 else ">50")
    )

    # Add the speaker metadata to the recordings
    test_recordings_df = test_recordings_df.merge(
        speaker_metadata, left_on="speaker_id", right_on="speaker_id"
    )
    train_recordings_df = train_recordings_df.merge(
        speaker_metadata, left_on="speaker_id", right_on="speaker_id"
    )

    # Create the dataset
    testset = Dataset.from_pandas(test_recordings_df).cast_column("filename", Audio())
    trainset = Dataset.from_dict(train_recordings_df).cast_column("filename", Audio())
    dataset_dict = DatasetDict({"train": trainset, "test": testset})

    # Create hub-id
    hub_id = f"{hub_id}_v{major_version}.{minor_version}"

    # Check if the dataset already exists on the hub, and if so, prompt the user to
    # update the version number
    api = HfApi()
    try:
        dataset_dict = api.dataset_info(repo_id=hub_id)
        if dataset_dict.id == hub_id:
            logger.error(
                (
                    f"The dataset {hub_id} already exists on the hub!",
                    "Please update the version number.",
                )
            )
            return
    except RepositoryNotFoundError:
        if major_version == 1 and minor_version == 0:
            logger.info(
                (
                    f"The dataset {hub_id} doesn't exist on the hub.",
                    "Creating a new dataset...",
                )
            )
        else:
            logger.error(
                (
                    f"The dataset {hub_id} doesn't exist on the hub!",
                    "Please create a new dataset with version 1.0.",
                )
            )
            return

    # Push the dataset to the hub
    while True:
        try:
            dataset_dict.push_to_hub(
                repo_id=hub_id,
                max_shard_size="500MB",
                private=private,
            )
            break
        except (RuntimeError, HTTPError) as e:
            logger.error(f"Error while pushing to hub: {e}")
            logger.info("Waiting a minute before trying again...")
            time.sleep(60)
            logger.info("Retrying...")


def timestamp(timestamp_str: str) -> datetime:
    """Convert a string representation of a timestamp to the timestamp.

    Args:
        timestamp_str:
            The string representation of the timestamp.

    Returns:
        The timestamp.
    """
    return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S+02:00")


if __name__ == "__main__":
    main()
