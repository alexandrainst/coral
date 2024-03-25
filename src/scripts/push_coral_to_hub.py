"""Pushes the processed CoRal dataset to the Hugging Face Hub.

Usage:
    python src/scripts/push_coral_to_hub.py\
            <path/to/recording/metadata/path> <path/to/speaker/metadata/path> <hub_id>
"""

import os
from pathlib import Path
from datasets import Dataset, Audio, DatasetDict
from datetime import datetime
from huggingface_hub import HfApi
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
    "--sentence_metadata_path",
    type=click.Path(exists=True),
    default="data/processed/sentences.xlsx",
    show_default=True,
    help="The path to the sentence metadata.",
)
@click.option(
    "--hub_id",
    type=str,
    default="CoRal-dataset/coral",
    show_default=True,
    help=(
        "The Hugging Face Hub id. Note that the version number will be appended to"
        " this id."
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
@click.option(
    "--max_num_conversation_recordings",
    type=int,
    default=3,
    show_default=True,
    help=(
        "The maximum number of conversation recordings to include in the validation"
        " set."
    ),
)
def main(
    recording_metadata_path: str | Path,
    speaker_metadata_path: str | Path,
    sentence_metadata_path: str | Path,
    hub_id: str,
    major_version: int,
    minor_version: int,
    private: bool,
    max_num_conversation_recordings: int,
) -> None:

    # Load the metadata and split into test/train speakers
    recording_metadata_path = Path(recording_metadata_path)
    recording_metadata = pd.read_excel(recording_metadata_path, index_col=0)

    # Load the sentence metadata
    sentence_metadata_path = Path(sentence_metadata_path)
    sentence_metadata = pd.read_excel(sentence_metadata_path, index_col=0)

    # Map sentence_id to text
    sentence_id_to_text = dict(
        zip(sentence_metadata["sentence_id"], sentence_metadata["text"])
    )
    recording_metadata["text"] = recording_metadata["sentence_id"].map(
        sentence_id_to_text
    )

    # Drop "sentence_id" column
    recording_metadata.drop(columns=["sentence_id"], inplace=True)

    # Change the dtype of all columns to string
    recording_metadata = recording_metadata.astype(str)

    # Recordings might have two speakers, so we split the speaker_id column into two
    recording_metadata[["speaker_id_1", "speaker_id_2"]] = recording_metadata[
        "speaker_id"
    ].str.split(",", expand=True)

    # Remove speaker_id column
    recording_metadata.drop(columns=["speaker_id"], inplace=True)

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
                for speaker_id in [row["speaker_id_1"], row["speaker_id_2"]]
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

    # Add the speaker metadata to the recordings. Since there possibly are two speakers
    # in each recording, we add the metadata for both speakers to the recordings. We
    # rename the columns to avoid conflicts.
    test_recordings_df = test_recordings_df.merge(
        speaker_metadata.rename(columns=lambda x: f"{x}_1"),
        on="speaker_id_1",
        how="left",
    )
    test_recordings_df = test_recordings_df.merge(
        speaker_metadata.rename(columns=lambda x: f"{x}_2"),
        on="speaker_id_2",
        how="left",
    )
    train_recordings_df = train_recordings_df.merge(
        speaker_metadata.rename(columns=lambda x: f"{x}_1"),
        on="speaker_id_1",
        how="left",
    )
    train_recordings_df = train_recordings_df.merge(
        speaker_metadata.rename(columns=lambda x: f"{x}_2"),
        on="speaker_id_2",
        how="left",
    )

    # Pick a validation set from the training set, by selecting a single recording
    # from each speaker. If make sure we do not select more than 3 conversation
    # recordings, as these are typically longer and might skew the validation set.
    validation_recordings = []
    for speaker_id in train_recordings_df["speaker_id_1"].unique():
        recording = train_recordings_df[
            train_recordings_df["speaker_id_1"] == speaker_id
        ].iloc[0]
        if (
            "conversation" in recording["filename"]
            and max_num_conversation_recordings > 0
        ):
            validation_recordings.append(recording)
            max_num_conversation_recordings -= 1
        elif "conversation" not in recording["filename"]:
            validation_recordings.append(recording)
    validation_recordings_df = pd.DataFrame(validation_recordings).astype(str)

    # Remove the validation recordings from the training set
    train_recordings_df = train_recordings_df[
        ~train_recordings_df["filename"].isin(validation_recordings_df["filename"])
    ].astype(str)

    # Remove conversation recordings from the training set
    train_read_aloud_df = (
        train_recordings_df[
            ~train_recordings_df["filename"].str.contains("conversation")
        ]
        .reset_index(drop=True)
        .astype(str)
    )
    test_read_aloud_df = (
        test_recordings_df[~test_recordings_df["filename"].str.contains("conversation")]
        .reset_index(drop=True)
        .astype(str)
    )
    validation_read_aloud_df = (
        validation_recordings_df[
            ~validation_recordings_df["filename"].str.contains("conversation")
        ]
        .reset_index(drop=True)
        .astype(str)
    )

    # Create the dataset
    testset = Dataset.from_pandas(test_read_aloud_df).cast_column("filename", Audio())
    trainset_read = Dataset.from_pandas(train_read_aloud_df).cast_column(
        "filename", Audio()
    )
    validationset = Dataset.from_dict(validation_read_aloud_df).cast_column(
        "filename",
        Audio(),
    )
    dataset_dict = DatasetDict(
        {
            "test": testset,
            "validation": validationset,
            "train_read_aloud": trainset_read,
        }
    )

    # Create hub-id
    if isinstance(major_version, int) and isinstance(minor_version, int):
        hub_id_v = f"{hub_id}_v{major_version}.{minor_version}"
    else:
        raise ValueError(
            (
                "The major and minor version numbers must be specified! Please update",
                " the version numbers.",
            )
        )

    # Check if the dataset already exists on the hub, and if so, prompt the user to
    # update the version number
    api = HfApi()

    # Get all datasets with "coral_v" in the name
    coral_datasets = api.list_datasets(search="coral_v")
    for dataset_info in coral_datasets:
        found_major, found_minor = dataset_info.id.split("_v")[1].split(".")
        if major_version < int(found_major) or (
            major_version == int(found_major) and minor_version < int(found_minor)
        ):
            raise ValueError(
                (
                    f"The dataset {hub_id_v} is of a lower version than the one on the",
                    " hub! Please update the version number.",
                )
            )
        elif major_version == int(found_major) and minor_version == int(found_minor):
            raise ValueError(
                (
                    f"The dataset {hub_id_v} already exists on the hub! Please update",
                    " the version number.",
                )
            )
    else:
        logger.info(f"The dataset {hub_id_v} doesn't exist on the hub. Proceeding...")

    # Push the dataset to the hub
    push_to_hub(dataset_dict, hub_id_v, private)


def push_to_hub(
    dataset_dict: DatasetDict,
    hub_id_v: str,
    private: bool,
) -> None:
    while True:
        try:
            dataset_dict.push_to_hub(
                repo_id=hub_id_v,
                max_shard_size="500 MB",
                token=os.environ["HF_TOKEN"],
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
