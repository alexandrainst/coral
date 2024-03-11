"""Pushes the processed CoRal dataset to the Hugging Face Hub.

Usage:
    python src/scripts/push_coral_to_hub.py\
            <path/to/recording/metadata/path> <path/to/speaker/metadata/path> <hub_id>
"""

from pathlib import Path
from datasets import Dataset, Audio, DatasetDict
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
    hub_id: str = "coral/iteration-1",
    private: bool = True,
) -> None:

    # Load the metadata and split into test/train speakers
    recording_metadata_path = Path(recording_metadata_path)
    recording_metadata = pd.read_excel(recording_metadata_path, index=0)
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
    speaker_metadata = pd.read_excel(speaker_metadata_path, index=0)

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
    speaker_metadata["age"] = speaker_metadata["age"].apply(
        lambda x: "<25" if x < 25 else "25-50" if x < 50 else ">50"
    )

    # Add the speaker metadata to the recordings
    test_recordings_df = test_recordings_df.merge(
        speaker_metadata, left_on="speaker_id", right_on="speaker_id"
    )
    train_recordings_df = train_recordings_df.merge(
        speaker_metadata, left_on="speaker_id", right_on="speaker_id"
    )

    # Create the dataset
    testset_dict = test_recordings_df.to_dict(orient="list")
    testset = Dataset.from_dict(testset_dict).cast_column("filename", Audio())
    trainset_dict = train_recordings_df.to_dict(orient="list")
    trainset = Dataset.from_dict(trainset_dict).cast_column("filename", Audio())
    dataset_dict = DatasetDict({"train": trainset, "test": testset})

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
