"""Pushes the processed CoRal dataset to the Hugging Face Hub.

Usage:
    python src/scripts/push_coral_to_hub.py\
            <path/to/recording/metadata/path> <path/to/speaker/metadata/path> <hub_id>
"""

from pathlib import Path
from datasets import Dataset, Audio, DatasetDict
from datetime import datetime
import pandas as pd
import click
import logging
import time

from requests import HTTPError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

test_speaker_ids = [
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
@click.argument("recording_metadata_path", type=click.Path(exists=True))
@click.argument("speaker_metadata_path", type=click.Path(exists=True))
@click.argument("hub_id", type=str)
@click.argument("major_version", type=int, default=1, show_default=True)
@click.argument("minor_version", type=int, default=0, show_default=True)
@click.option(
    "--private",
    is_flag=True,
    default=True,
    show_default=True,
    help="Whether to make the dataset private on the Hugging Face Hub.",
)
def main(
    recording_metadata_path: Path | str = Path("data/processed/recordings.xlsx"),
    speaker_metadata_path: Path | str = Path("data/hidden/speakers.xlsx"),
    hub_id: str = "alexandrainst/coral",
    major_version: int = 1,
    minor_version: int = 0,
    private: bool = True,
) -> None:

    # Load the metadata and split into test/train speakers
    recording_metadata_path = Path(recording_metadata_path)
    recording_metadata = pd.read_excel(recording_metadata_path, index_col=0)

    # Change the dtype of all columns to string
    recording_metadata = recording_metadata.astype(str)

    # All recordings which were made before 2024-03-11 are defined as iteration 1
    # and anything after that is defined as iteration 2
    iteration_periods: dict[str, tuple[datetime, datetime]] = {
        "iteration_1": (datetime.min, timestamp("2024-03-15T00:00:00+02:00")),
        "iteration_2": (timestamp("2024-03-15T00:00:00+02:00"), datetime.max),
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
            
        for iteration_name, (iteration_start, iteration_end) in iteration_periods.items():
            if start_time >= iteration_start and start_time < iteration_end:
                return iteration_name
        else:
            raise ValueError(f"The start time {start_time} doesn't correspond to any iteration!")

    recording_metadata["iteration"] = recording_metadata["start"].apply(get_iteration_name)

    test_recordings = []
    train_recordings = []
    for _, row in recording_metadata.iterrows():
        if any(
            [
                speaker_id in test_speaker_ids
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


def timestamp(timestamp) -> datetime:
    return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S+02:00")


if __name__ == "__main__":
    main()
