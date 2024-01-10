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
    """Selects speakers such that the selection has 15% from each region.

    If the current selection already has 15% from each region, the selection
    is returned as is. If the current selection has a region which account for less
    than 15%, then the remaining regions are reduced such that the smallest region
    accounts for 15% of the selection.

    Args:
        current_selection (pd.DataFrame):
            The current selection of speakers.

    Returns:
        pd.DataFrame:
            The updated selection of speakers.
    """

    # Check if all of the regions are above the threshold
    threshold = 0.15 * current_selection.length.sum()
    regions = [
        (region, region.length.sum())
        for _, region in current_selection.groupby("region")
    ]
    if all([threshold < length for _, length in regions]):
        return current_selection

    # Define new threshold as the size of the smallest region
    regions_dist = sorted(regions, key=lambda x: x[1])

    # Get the length of speakers from smallest regions
    least_repr_region_length = regions_dist[0][1]

    # We remove speakers from the overrepresented regions, uniformly from each
    # regions.
    new_total_length = least_repr_region_length / 0.15
    new_threshold = new_total_length * (0.85 / 4)  # 4 remaining regions

    # Remove recordings from the regions until they are below the threshold.
    # We remove the recordings with the longest duration first.
    # This is done to ensure that we remove the least amount of speakers.
    deselected_speakers = []
    for region, length in regions:
        for _, row in region.sort_values(by="length", ascending=False).iterrows():
            # Check if removing the speaker will bring the selection below the threshold
            if length >= new_threshold:
                length -= row.length
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

    If the current selection already has 10% speakers with accent, the selection
    is returned as is. If the current selection has kess than 10% speakers with accent,
    then the amount of speakers without accent is reduced such that speakers with
    accent accounts for 10% of the selection.

    Args:
        current_selection (pd.DataFrame):
            The current selection of speakers.

    Returns:
        pd.DataFrame:
            The updated selection of speakers.
    """

    # Check if all the selection has more than 10% with accent
    accent_dist = []
    for _, accent in current_selection.groupby("accent"):
        length = accent.length.sum()
        accent_dist.append((accent, length))

    threshold = 0.1 * current_selection.length.sum()
    if all([threshold < length for _, length in accent_dist]):
        return current_selection

    # Either there are too few speakers with accent, or too many. We remove
    # speakers accordingly.
    accent_dist = sorted(accent_dist, key=lambda x: x[1])

    # Get the length of speakers with and without accent
    least_repr_accent_group_length = accent_dist[0][1]
    most_repr_accent_group = accent_dist[1][0]
    most_repr_accent_group_length = accent_dist[1][1]

    # We remove speakers without accent until we have 10% speakers with accent.
    # This is done by adjusting the threshold.
    new_total_length = least_repr_accent_group_length / 0.1
    new_threshold = new_total_length * 0.9
    region_groups = most_repr_accent_group.groupby("region")
    deselected_speakers = []
    index = 0

    # We wish to remove speakers with accent uniformly over regions, but
    # when we adjusted to have the correct amount of data from each of the
    # regions, we removed the speakers with the longest recordings first from the
    # overrepresented regions. This means that the region which started out being
    # most underrepresented now still has the speakers with the longest recordings.
    # Hence when we remove speakers from this region, we are removing far more data
    # than when we remove speakers from the other regions. This is not ideal, as we
    # will skew the region distribution. Hence we skip the smallest region when
    # removing speakers with accent.
    smallest_region = (
        current_selection.groupby("region").length.sum().sort_values().keys()[0]
    )
    while most_repr_accent_group_length > new_threshold:
        # Remove index-th speaker from each region, as long as the selection
        # is above the threshold
        for _, region in region_groups:
            is_smallest_region = region.region.values[0] == smallest_region
            no_more_speakers_to_remove = region.shape[0] <= index
            group_is_a_single_speaker = region.shape[0] <= 1
            if (
                no_more_speakers_to_remove
                or group_is_a_single_speaker
                or is_smallest_region
            ):
                continue

            row = region.sort_values(by="length", ascending=False).iloc[index]

            # Check if removing the speaker will bring the selection below the
            # threshold
            if most_repr_accent_group_length > new_threshold:
                most_repr_accent_group_length -= row.length
                deselected_speakers.append(row.speaker_id)
            else:
                break
        index += 1

    # Update the current selection
    current_selection = current_selection[
        ~current_selection.speaker_id.isin(deselected_speakers)
    ]

    return current_selection


def select_by_gender(
    current_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Remove speakers such that each gender is represented 45%.

    Args:
        current_selection (pd.DataFrame):
            The current selection of speakers.

    Returns:
        pd.DataFrame:
            The updated selection of speakers.
    """

    # Check if all the selection has more than 45% of each gender
    deselected_speakers = []
    gender_dist = []
    for _, gender in current_selection.groupby("gender"):
        if gender.gender.values[0] in ["male", "female"]:
            gender_dist.append((gender, gender.length.sum()))
        else:
            deselected_speakers.extend(gender.speaker_id.values.tolist())

    threshold = 0.45 * current_selection.length.sum()
    if all([threshold < length for _, length in gender_dist]):
        return current_selection

    # Sort by length
    gender_dist = sorted(gender_dist, key=lambda x: x[1])

    # Get the lengths to create new thresholds.
    least_repr_gender_length = gender_dist[0][1]
    most_repr_gender = gender_dist[1][0]
    most_repr_gender_length = gender_dist[1][1]

    # We remove speakers from the overrepresented gender until we have 45%
    # of each gender. We remove speakers uniformly from each region and accent.
    new_total_length = least_repr_gender_length / 0.45
    new_threshold = new_total_length * 0.55
    region_accent_groups = most_repr_gender.groupby(["region", "accent"])
    index = 0
    while most_repr_gender_length > new_threshold:
        # Remove index-th speaker from each region, as long as the selection
        # is above the threshold
        for _, region_accent in region_accent_groups:
            no_more_speakers_to_remove = region_accent.shape[0] <= index
            group_is_a_single_speaker = region_accent.shape[0] <= 1
            if no_more_speakers_to_remove or group_is_a_single_speaker:
                continue

            row = region_accent.sort_values(by="length", ascending=False).iloc[index]

            # Check if removing the speaker will bring the selection below the
            # threshold
            if most_repr_gender_length > new_threshold:
                most_repr_gender_length -= row.length
                deselected_speakers.append(row.speaker_id)
            else:
                break
        index += 1

    # Update the current selection
    current_selection = current_selection[
        ~current_selection.speaker_id.isin(deselected_speakers)
    ]

    return current_selection


def select_by_age(
    current_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Remove speakers such that each age group is represented 20%.

    If the current selection already has 20% speakers of each age group, the selection
    is returned as is. If the current selection has less than this, then the amount of
    speakers from the overrepresented age groups are reduced such that each age group
    accounts for 20% of the selection.
    Args:
        current_selection (pd.DataFrame):
            The current selection of speakers.

    Returns:
        pd.DataFrame:
            The updated selection of speakers.
    """
    # Sort by length and take the age group with the least amount of recordings
    # as the threshold.
    age_groups = [
        (age_group, age_group.length.sum())
        for _, age_group in current_selection.groupby("age")
    ]

    # Check if all the selection has more than 20% of each age group
    threshold = 0.2 * current_selection.length.sum()
    if all([threshold < length for _, length in age_groups]):
        return current_selection

    # Define new threshold as the size of the smallest age group
    age_group_dist = sorted(age_groups, key=lambda x: x[1])

    # Get the length of speakers from smallest age group
    least_repr_age_group_length = age_group_dist[0][1]

    # We remove speakers from the overrepresented age groups, uniformly over
    # regions, accent and gender.
    new_total_length = least_repr_age_group_length / 0.1
    new_threshold = new_total_length * (0.90 / 2)  # 2 remaining age groups
    deselected_speakers = []
    for age_group, length in age_groups:
        if length <= new_threshold:
            continue
        else:
            region_accent_gender_groups = age_group.groupby(
                ["region", "accent", "gender"]
            )
            index = 0
            while length > new_threshold:
                # Remove index-th speaker from each region, as long as the selection
                # is above the threshold
                for _, region_accent_gender in region_accent_gender_groups:
                    no_more_speakers_to_remove = region_accent_gender.shape[0] <= index
                    group_is_a_single_speaker = region_accent_gender.shape[0] <= 1
                    if no_more_speakers_to_remove or group_is_a_single_speaker:
                        continue

                    row = region_accent_gender.sort_values(
                        by="length", ascending=False
                    ).iloc[index]

                    # Check if removing the speaker will bring the selection below the
                    # threshold
                    if length > new_threshold:
                        length -= row.length
                        deselected_speakers.append(row.speaker_id)
                    else:
                        break
                index += 1

    # Update the current selection
    current_selection = current_selection[
        ~current_selection.speaker_id.isin(deselected_speakers)
    ]

    return current_selection


def select_by_length(current_selection: pd.DataFrame) -> pd.DataFrame:
    """Selects speakers based on the length of their recordings.

    Args:
        current_selection (pd.DataFrame):
            The current selection of speakers.

    Returns:
        pd.DataFrame:
            The updated selection of speakers.
    """
    # We need to keep as much of the conversation recordings as possible, as
    # these are the most important for the test set. Hence we remove speakers
    # with the longest read aloud recordings first, uniformly over regions, accent
    # gender and the older age groups.

    threshold = 7.5 * 3600  # 7.5 hours
    mid_old = current_selection[~(current_selection.age == "<25")]
    current_length = mid_old.read_aloud_length.sum()
    index = 0
    deselected_speakers = []
    while current_length > threshold:
        for _, group in mid_old.groupby(["age", "gender", "accent", "region"]):
            sorted_group = group.sort_values(by="read_aloud_length", ascending=False)
            # Check if there are any speakers left in the group
            if len(sorted_group) <= 1 or len(sorted_group) <= index:
                continue

            # Check if we are removing conversation data.
            row = sorted_group.iloc[index]
            has_conversation_data = row.conversation_length != 0.0
            if has_conversation_data:
                continue
            if current_length > threshold:
                current_length -= row.read_aloud_length
                deselected_speakers.append(row.speaker_id)
            else:
                break
        index += 1

    # Update the current selection
    current_selection = current_selection[
        ~current_selection.speaker_id.isin(deselected_speakers)
    ]
    return current_selection


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
    for speaker_id in tqdm(speaker_metadata.speaker_id):
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

        dialect = speaker_metadata[
            speaker_metadata.speaker_id == speaker_id
        ].dialect.values[0]

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
                "dialect": dialect,
                "conversation_length": conversation_length,
                "read_aloud_length": read_aloud_length,
                "length": conversation_length + read_aloud_length,
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
    current_selection = select_by_accent(
        current_selection=current_selection,
    )
    current_selection = select_by_gender(
        current_selection=current_selection,
    )
    current_selection = select_by_age(
        current_selection=current_selection,
    )

    # Select certain speakers to fix age, conversation/read aloud distribution
    hand_picked_speakers = []
    young = speaker_selection_criteria_data[
        speaker_selection_criteria_data.age == "<25"
    ]

    # Select young male with the longest conversation recording
    young_males = young[young.gender == "male"]
    hand_picked_speakers.append(
        young_males.sort_values(
            by="conversation_length", ascending=False
        ).speaker_id.values[0]
    )

    # Select the young person with the longest conversation recording

    hand_picked_speakers.extend(
        young.sort_values(by="conversation_length", ascending=False).speaker_id.values[
            :1
        ]
    )

    current_selection = pd.concat(
        [
            current_selection,
            speaker_selection_criteria_data[
                speaker_selection_criteria_data.speaker_id.isin(hand_picked_speakers)
            ],
        ]
    )

    # Select speakers such that the test set has the correct distribution of
    # conversation/read aloud distribution

    current_selection = select_by_length(
        current_selection=current_selection,
    )
    current_selection.to_csv(processed_data_path / "test_speakers.csv", index=False)


if __name__ == "__main__":
    main()
