"""Script for selecting a test set from the full dataset.

Usage:
    python select_testset.py +datasets=coral_test_set
"""

import hydra
import contextlib
import wave
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


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


def select_by_region(
    current_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Selects speakers such that the selection has 10% from each region.

    Args:
        current_selection (pd.DataFrame):
            The current selection of speakers.

    Returns:
        pd.DataFrame:
            The updated selection of speakers.
    """

    # Remove speakers from the current selection if their region is
    # already represented with more than 10% of the total recordings
    total_recordings = (
        current_selection.conversation_length.sum()
        + current_selection.read_aloud_length.sum()
    )
    threshold = 0.1 * total_recordings

    # Get the regions that are already represented with more than 10%
    # of the total recordings
    regions = []
    for _, region in current_selection.groupby("region"):
        length = region.conversation_length.sum() + region.read_aloud_length.sum()
        if length > threshold:
            regions.append((region, length))

    # Remove recordings from the regions that are already represented
    # with more than 10% of the total recordings
    deselected_speakers = []
    for region, length in regions:
        for _, row in region.sort_values(
            by="read_aloud_length", ascending=False
        ).iterrows():
            # Check if removing the speaker will bring the selection below the threshold
            recording_length = row.conversation_length + row.read_aloud_length
            if length - recording_length > threshold:
                length -= recording_length
                deselected_speakers.append(row.speaker_id)
        else:
            break

    # Update the current selection
    current_selection = current_selection[
        ~current_selection.speaker_id.isin(deselected_speakers)
    ]

    return current_selection


# TODO: Just copied from select_by_region!!!!
def select_by_gender(
    current_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Remove speakers such that the selection has 10% from each region.

    Args:
        current_selection (pd.DataFrame):
            The current selection of speakers.

    Returns:
        pd.DataFrame:
            The updated selection of speakers.
    """

    # Remove speakers from the current such that the selection has 10% speakers
    # with accent and 90% without accent
    total_recordings = (
        current_selection.conversation_length.sum()
        + current_selection.read_aloud_length.sum()
    )
    threshold = 0.1 * total_recordings

    # Get the distribution of speakers with and without accent
    accent_dist = []
    for _, accent in current_selection.groupby("accent"):
        length = accent.conversation_length.sum() + accent.read_aloud_length.sum()
        accent_dist.append((accent, length))

    # Sort by length
    accent_dist = sorted(accent_dist, key=lambda x: x[1], reverse=True)

    # Too many speakers with accent
    has_accent = accent_dist[0][0]
    has_accent_length = accent_dist[0][1]
    does_not_have_accent = accent_dist[1][0]
    does_not_have_accent_length = accent_dist[1][1]

    # Too many speakers with accent
    if has_accent_length > threshold:
        # We remove speakers with accent until we have 10% speakers with accent
        # and 90% without accent, this is done by scaling the threshold
        new_total_length = does_not_have_accent_length / (1 - threshold)
        new_threshold = new_total_length * threshold

        # Remove speakers with accent
        deselected_speakers = []
        for _, row in has_accent.sort_values(
            by="read_aloud_length", ascending=False
        ).iterrows():
            # Check if removing the speaker will bring the selection below the threshold
            recording_length = row.conversation_length + row.read_aloud_length
            if has_accent_length - recording_length > new_threshold:
                has_accent_length -= recording_length
                deselected_speakers.append(row.speaker_id)
            else:
                break

    # Too many speakers without accent
    elif does_not_have_accent_length > 1 - threshold:
        # We remove speakers without accent until we have 10% speakers with accent
        # and 90% without accent, this is done by scaling the threshold
        new_total_length = has_accent_length / threshold
        new_threshold = new_total_length * (1 - threshold)

        # Remove speakers without accent
        deselected_speakers = []
        for _, row in does_not_have_accent.sort_values(
            by="read_aloud_length", ascending=False
        ).iterrows():
            # Check if removing the speaker will bring the selection below the threshold
            recording_length = row.conversation_length + row.read_aloud_length
            if does_not_have_accent_length - recording_length > new_threshold:
                does_not_have_accent_length -= recording_length
                deselected_speakers.append(row.speaker_id)
            else:
                break

    # Update the current selection
    current_selection = current_selection[
        ~current_selection.speaker_id.isin(deselected_speakers)
    ]

    return current_selection


def select_by_accent(
    current_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Remove speakers such 10% of speakers have accent and 90% do not.

    Args:
        current_selection (pd.DataFrame):
            The current selection of speakers.

    Returns:
        pd.DataFrame:
            The updated selection of speakers.
    """

    # Remove speakers from the current such that the selection has 10% speakers
    # with accent and 90% without accent
    total_recordings = (
        current_selection.conversation_length.sum()
        + current_selection.read_aloud_length.sum()
    )
    threshold = 0.1 * total_recordings

    # Get the distribution of speakers with and without accent
    accent_dist = []
    for _, accent in current_selection.groupby("accent"):
        length = accent.conversation_length.sum() + accent.read_aloud_length.sum()
        accent_dist.append((accent, length))

    # Sort by length
    accent_dist = sorted(accent_dist, key=lambda x: x[1], reverse=True)

    # Too many speakers with accent
    has_accent = accent_dist[0][0]
    has_accent_length = accent_dist[0][1]
    does_not_have_accent = accent_dist[1][0]
    does_not_have_accent_length = accent_dist[1][1]

    # Too many speakers with accent
    if has_accent_length > threshold:
        # We remove speakers with accent until we have 10% speakers with accent
        # and 90% without accent, this is done by scaling the threshold
        new_total_length = does_not_have_accent_length / (1 - threshold)
        new_threshold = new_total_length * threshold

        # Remove speakers with accent, uniformly from each region
        deselected_speakers = []
        region_groups = has_accent.groupby("region")
        index = 0
        while has_accent_length > new_threshold:
            # Remove index-th speaker from each region, as long as the selection
            # is above the threshold
            for _, region in region_groups:
                row = region.sort_values(by="read_aloud_length", ascending=False).iloc[
                    index
                ]

                # Check if removing the speaker will bring the selection below the
                # threshold
                recording_length = row.conversation_length + row.read_aloud_length
                if has_accent_length > new_threshold:
                    has_accent_length -= recording_length
                    deselected_speakers.append(row.speaker_id)
                else:
                    break

    # Too many speakers without accent
    elif does_not_have_accent_length > 1 - threshold:
        # We remove speakers without accent until we have 10% speakers with accent
        # and 90% without accent, this is done by scaling the threshold
        new_total_length = has_accent_length / threshold
        new_threshold = new_total_length * (1 - threshold)

        # Remove speakers without accent, uniformly from each region
        deselected_speakers = []
        region_groups = does_not_have_accent.groupby("region")
        index = 0
        while does_not_have_accent_length > new_threshold:
            # Remove index-th speaker from each region, as long as the selection
            # is above the threshold
            for _, region in region_groups:
                row = region.sort_values(by="read_aloud_length", ascending=False).iloc[
                    index
                ]

                # Check if removing the speaker will bring the selection below the
                # threshold
                recording_length = row.conversation_length + row.read_aloud_length
                if has_accent_length > new_threshold:
                    has_accent_length -= recording_length
                    deselected_speakers.append(row.speaker_id)
                else:
                    break

    # Update the current selection
    current_selection = current_selection[
        ~current_selection.speaker_id.isin(deselected_speakers)
    ]

    return current_selection


def select_by_length(
    current_selection: pd.DataFrame, type_of_recording: str
) -> pd.DataFrame:
    """Selects speakers based on the length of their recordings.

    Args:
        current_selection (pd.DataFrame):
            The current selection of speakers.
        type_of_recording (str):
            The type of recordings to select on.
            Either `conversation_length` or `read_aloud_length`.

    Returns:
        pd.DataFrame:
            The updated selection of speakers.
    """
    pass


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
        age = (
            speaker_metadata[speaker_metadata.speaker_id == speaker_id]
            .age.apply(lambda x: "<25" if x < 25 else "25-50" if x < 50 else ">50")
            .values[0]
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
    speaker_selection_criteria_data = pd.DataFrame(speaker_selection_criteria)

    # We start with the full dataset and then remove speakers such that the
    # selection has the correct distribution of speakers for each of the selection
    # criteria. The order of the selection criteria is important, as the first
    # criteria will be the most important, and the last criteria will be the
    # least important. The order of the selection criteria is as follows:
    # 1. Region
    # 2. Accent
    # 3. Gender
    # 4. Age
    # Once we've selected a subset of speakers based on the selection criteria,
    # we deselect speakers uniformly from each constallation of the selection
    # criteria, until we reach the desired test set size.

    current_selection = speaker_selection_criteria_data

    current_selection = select_by_region(
        current_selection=current_selection,
    )
    current_selection = select_by_gender(
        current_selection=current_selection,
    )
    current_selection = select_by_length(
        current_selection=current_selection,
        type_of_recording="conversation_length",
    )


if __name__ == "__main__":
    main()
