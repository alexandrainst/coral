"""Script for selecting a test set from the full dataset.

Usage:
    python select_testset.py +datasets=coral_test_set
"""

import hydra
import contextlib
import wave
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig


def get_recordings_length(recordings: pd.DataFrame) -> tuple[float, float]:
    """Returns the length of a recording in seconds.

    Args:
        recordings (pd.DataFrame):
            The dataframe with recording metadata.

    Returns:
        tuple[float,float]:
            The length of the conversation and the read aloud recordings resp. in
            seconds.
    """
    conversation_length = 0.0
    read_aloud_length = 0.0
    for _, row in recordings.iterrows():
        with contextlib.closing(wave.open(row.filename, "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        if "conversation" in row.filename:
            conversation_length += duration
        else:
            read_aloud_length += duration
    return conversation_length, read_aloud_length


def select_by_conversation(current_selection: pd.DataFrame) -> pd.DataFrame:
    """Selects speakers based on the length of their conversation recordings.

    Args:
        current_selection (pd.DataFrame):
            The current selection of speakers.

    Returns:
        pd.DataFrame:
            The updated selection of speakers.
    """

    # Sort by length and select speakers with the shortest recordings
    # such that their their conversation recordings are 7.5 hours each.
    cumulated_conversation_length = 0
    selected_speakers = []
    for i, row in current_selection.sort_values(
        by=["conversation_length", "read_aloud_length"], ascending=True
    ).iterrows():
        if cumulated_conversation_length + row.conversation_length < 7.5 * 3600:
            cumulated_conversation_length += row.conversation_length
            selected_speakers.append(row.speaker_id)

    # Update the current selection
    current_selection = current_selection[
        current_selection.speaker_id.isin(selected_speakers)
    ]


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Selects the test set from the full CoRal dataset."""
    data_dir = Path(cfg.dirs.data)
    processed_data_path = data_dir / cfg.dirs.processed
    speaker_metadata_path = data_dir / cfg.dirs.hidden / "speakers.xlsx"
    processed_audio_path = processed_data_path / "processed_audio"
    recordings_metadata_path = processed_data_path / "recordings.xlsx"

    if not processed_audio_path.exists():
        raise FileNotFoundError(f"{processed_audio_path} does not exist")
    if not speaker_metadata_path.exists():
        raise FileNotFoundError(f"{speaker_metadata_path} does not exist")
    if not recordings_metadata_path.exists():
        raise FileNotFoundError(f"{recordings_metadata_path} does not exist")

    # Load the metadata
    speaker_metadata = pd.read_excel(speaker_metadata_path)
    recordings_metadata = pd.read_excel(recordings_metadata_path)

    # Load dialect to region mapping
    if not cfg.datasets.coral_test_set.region_to_dialect:
        raise FileNotFoundError(
            (
                "`coral_test_set` was not found in the config, "
                "please add under `defaults.datasets`"
            )
        )
    region_to_dialect = cfg.datasets.coral_test_set.region_to_dialect
    dialect_to_region = {
        dialect: region
        for region, dialects in region_to_dialect.items()
        for dialect in dialects
    }

    # Loop over the speakers to get selection criteria
    speaker_selection_criteria = []
    for speaker_id in tqdm(speaker_metadata.iloc[:10, :].speaker_id):
        # Get the recordings of the speaker
        recordings = recordings_metadata[
            recordings_metadata.speaker_id.str.split(",").apply(
                lambda ids: speaker_id in ids
            )
        ]

        # Get length of recordings
        conversation_length, read_aloud_length = get_recordings_length(recordings)

        # Skip if the speaker there is no recordings
        if conversation_length == 0 and read_aloud_length == 0:
            continue

        # Get the region of the speaker, default to sjælland
        region = dialect_to_region.get(
            speaker_metadata[speaker_metadata.speaker_id == speaker_id]
            .dialect.values[0]
            .lower(),
            "sjælland",
        )

        # Get the gender of the speaker
        gender = speaker_metadata[
            speaker_metadata.speaker_id == speaker_id
        ].gender.values[0]

        # Get the age of the speaker
        age = speaker_metadata[speaker_metadata.speaker_id == speaker_id].age.apply(
            lambda x: "<25" if x < 25 else "25-50" if x < 50 else ">50"
        )

        # Get the accent of the speaker
        accent = (
            speaker_metadata[speaker_metadata.speaker_id == speaker_id]
            .birthplace.apply(lambda x: 0 if x in ["Danmark", "Denmark"] else 1)
            .values[0]
        )

        # Append to the selection criteria
        speaker_selection_criteria.append(
            {
                "speaker_id": speaker_id,
                "region": region,
                "gender": gender,
                "age": age,
                "accent": accent,
                "conversation_length": conversation_length,
                "read_aloud_length": read_aloud_length,
            }
        )

    # Convert to dataframe
    speaker_selection_criteria = pd.DataFrame(speaker_selection_criteria)

    # Load the criteria selection priority
    selection_priority = cfg.datasets.coral_test_set.selection_criteria_priority

    current_selection = speaker_selection_criteria
    for criterion in selection_priority:
        if criterion == "conversation_length":
            select_by_conversation(current_selection)


if __name__ == "__main__":
    main()
