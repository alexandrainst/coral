"""Get the speaker IDs for the CoRal test and validation splits.

The test set is subject to the following constraints:
    - At least 7.5 hours
    - At least 40% of the test set must be of each gender
    - At least 20% of the test set must be of each age group (0-24, 25-49, 50+)
    - At least 10% of the test set must be of each dialect group
    - At least 10% of the test set must be of speakers with a non-native accent

The validation split has no formal criteria, but must be significantly smaller than the
test set and should have roughly the same distribution.

These constraints, along with all the other hyperparameters related to the creation of
the splits, are defined in the `dataset_creation` configuration file.

Developers:
    - Oliver Kinch (oliver.kinch@alexandra.dk)
    - Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)

Usage:
    python src/scripts/get_coral_split_ids.py <key>=<value> <key>=<value> ...
"""

import logging
import warnings
from pathlib import Path
from typing import NamedTuple

import hydra
import numpy as np
import pandas as pd
from datasets import (
    DatasetDict,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
)
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pandas.errors import SettingWithCopyWarning
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("get_coral_split_ids")

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(config_path="../../config", config_name="split_creation", version_base=None)
def main(config: DictConfig) -> None:
    """Main function to get the speaker IDs for the CoRal test and validation splits.

    Args:
        config:
            The Hydra configuration object
    """
    mean_seconds_per_sample = config.mean_seconds_per_sample
    num_attempts = config.num_split_attempts
    df = load_coral_metadata_df(
        sub_dialect_to_dialect=config.sub_dialect_to_dialect,
        age_groups=config.age_groups,
        max_cer=config.requirements.max_cer,
        streaming=config.streaming,
        dataset_path=config.dataset_path,
        revision=config.dataset_revision,
        cache_dir=config.cache_dir,
    )
    logger.info(f"Loaded processed CoRal metadata with {len(df):,} samples.")

    def compute_test_candidate(
        seed: int,
        requirements: dict[str, float],
        banned_speakers: set[str],
        min_hours: float,
        max_hours: float,
    ) -> EvalDataset | None:
        candidate = EvalDataset(
            df=df,
            min_samples=int(min_hours * 60 * 60 / mean_seconds_per_sample),
            max_samples=int(max_hours * 60 * 60 / mean_seconds_per_sample),
            requirements=requirements,
            banned_speakers=banned_speakers,
            seed=seed,
            genders=config.genders,
            dialects=config.dialects,
            age_groups=config.age_groups,
            mean_seconds_per_sample=mean_seconds_per_sample,
        )
        if not candidate.satisfies_requirements:
            return None
        return candidate

    # Build test split
    with Parallel(n_jobs=-1, batch_size=10) as parallel:
        test_candidates = parallel(
            delayed(function=compute_test_candidate)(
                seed=seed,
                requirements=dict(
                    gender=config.requirements.test.gender_pct,
                    dialect=config.requirements.test.dialect_pct,
                    age_group=config.requirements.test.age_group_pct,
                ),
                banned_speakers=set(),
                min_hours=config.requirements.test.min_hours,
                max_hours=config.requirements.test.max_hours,
            )
            for seed in tqdm(
                range(4242, 4242 + num_attempts), desc="Computing test splits"
            )
        )
    test_candidates = [
        candidate for candidate in test_candidates if candidate is not None
    ]

    if not test_candidates:
        raise ValueError("No test candidate satisfy the requirements!")

    # Pick the test dataset that is both short and difficult
    difficulty_sorted_candidates = sorted(
        test_candidates, key=lambda x: x.difficulty, reverse=True
    )
    length_sorted_candidates = sorted(test_candidates, key=len)
    candidate_scores = {
        candidate: difficulty_sorted_candidates.index(candidate)
        + length_sorted_candidates.index(candidate)
        for candidate in test_candidates
    }
    test_dataset = min(test_candidates, key=lambda x: candidate_scores[x])
    logger.info(f"Test dataset:\n{test_dataset}")

    # Build validation split
    with Parallel(n_jobs=-1, batch_size=10) as parallel:
        val_candidates = parallel(
            delayed(function=compute_test_candidate)(
                seed=seed,
                requirements=dict(
                    gender=config.requirements.val.gender_pct,
                    dialect=config.requirements.val.dialect_pct,
                    age_group=config.requirements.val.age_group_pct,
                ),
                banned_speakers=test_dataset.speakers,
                min_hours=config.requirements.val.min_hours,
                max_hours=config.requirements.val.max_hours,
            )
            for seed in tqdm(
                range(4242, 4242 + num_attempts), desc="Computing val splits"
            )
        )

    val_candidates = [
        candidate for candidate in val_candidates if candidate is not None
    ]

    if not val_candidates:
        raise ValueError("No validation candidate satisfy the requirements!")

    # Pick the test dataset that is both short and difficult
    difficulty_sorted_candidates = sorted(
        val_candidates, key=lambda x: x.difficulty, reverse=True
    )
    length_sorted_candidates = sorted(val_candidates, key=len)
    candidate_scores = {
        candidate: difficulty_sorted_candidates.index(candidate)
        + length_sorted_candidates.index(candidate)
        for candidate in val_candidates
    }
    val_dataset = min(val_candidates, key=lambda x: candidate_scores[x])
    logger.info(f"Validation dataset:\n{val_dataset}")

    assert set(test_dataset.speakers).intersection(val_dataset.speakers) == set()


class AgeGroup(NamedTuple):
    """Named tuple to represent an age group."""

    min: int
    max: int | None

    def __repr__(self) -> str:
        """Return the string representation of the AgeGroup class."""
        if self.max is None:
            return f"{self.min}-"
        return f"{self.min}-{self.max - 1}"

    def __contains__(self, age: object) -> bool:
        """Check if an age is in the age group.

        Args:
            age:
                The age to check.

        Returns:
            Whether the age is in the age group.
        """
        if not isinstance(age, int):
            return False
        return self.min <= age and (self.max is None or age < self.max)


class EvalDataset:
    """Dataset class to keep track of the samples in the dataset.

    Attributes:
        df (pd.DataFrame):
            Dataframe of the Coral dataset.
        min_samples (int):
            The minimum amount of samples in the dataset.
        max_samples (int):
            The maximum amount of samples in the dataset.
        requirements (dict):
            The requirements for the dataset.
        banned_speakers (set[str]):
            Set of speaker IDs that should not be included in the dataset.
        indices (list[int]):
            List of indices of the Coral dataset that will be included in the dataset.
        speakers (set[str]):
            List of speaker IDs that will be included in the dataset.
        rng (np.random.Generator):
            Random number generator.
        mean_seconds_per_sample (float):
            The mean duration of a sample in seconds. Only used for logging.
        counts (dict):
            Count of each feature in the dataset.
        weights (dict):
            Weights of each feature in the dataset.
        satisfies_requirements (bool):
            If the dataset satisfies all the requirements.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        min_samples: int,
        max_samples: int,
        requirements: dict[str, float],
        banned_speakers: set[str],
        seed: int,
        genders: list[str],
        dialects: list[str],
        age_groups: list[tuple[int, int]],
        mean_seconds_per_sample: float,
    ) -> None:
        """Initialise the Dataset class.

        Args:
            df:
                Dataframe of the Coral dataset.
            min_samples:
                The minimum amount of samples in the dataset.
            max_samples:
                The maximum amount of samples in the dataset
            requirements:
                The requirements for the dataset.
            banned_speakers:
                Set of speaker IDs that should not be included in the dataset.
            seed:
                The seed for the random number generator.
            genders:
                A list of possible gender values.
            dialects:
                A list of possible dialect values.
            age_groups:
                A list of tuples with the minimum and maximum age of each age group.
            mean_seconds_per_sample:
                The mean duration of a sample in seconds. Only used for logging.
        """
        self.df = df
        self.min_samples: int = min_samples
        self.max_samples: int = max_samples
        self.requirements: dict[str, float] = requirements
        self.banned_speakers: set[str] = banned_speakers
        self.indices: list[int] = list()
        self.speakers: set[str] = set()
        self.rng = np.random.default_rng(seed=seed)
        self.mean_seconds_per_sample = mean_seconds_per_sample
        self.age_groups = [
            AgeGroup(min=min_age, max=max_age) for min_age, max_age in age_groups
        ]
        self.counts: dict[str, dict[str, int]] = dict(
            gender={gender: 0 for gender in genders},
            dialect={dialect: 0 for dialect in dialects},
            age_group={str(age_group): 0 for age_group in self.age_groups},
        )
        self.weights: dict[str, dict[str, float]] = {
            key: self._compute_weights(count=count, min_value=self.requirements[key])
            for key, count in self.counts.items()
        }
        self.satisfies_requirements = True
        self.populate()

    @property
    def difficulty(self) -> float:
        """Return the difficulty of the dataset."""
        return self.df.loc[self.indices].asr_cer.mean()

    def add_speaker_samples(self, speaker: str) -> "EvalDataset":
        """Add all samples of a speaker to the dataset.

        Args:
            speaker:
                The id of the speaker
        """
        self.speakers.add(speaker)

        speaker_samples = self.df.query("id_speaker == @speaker")
        n_samples = len(speaker_samples)
        indices = speaker_samples.index.tolist()
        self.indices.extend(indices)

        # Assuming that all samples of a speaker have the same gender, dialect,
        # age_group, and native_language
        row = speaker_samples.iloc[0]
        for key, count in self.counts.items():
            count[row[key]] += n_samples

        self._update_weights()

        return self

    def _give_score(self, row: pd.Series) -> float:
        """Return a score of a speaker in a row."""
        return sum(
            weight[row[key]]  # type: ignore[index]
            for key, weight in self.weights.items()
        )

    def populate(self) -> "EvalDataset":
        """Populate the dataset with samples.

        Returns:
            EvalDataset object with samples.
        """
        df_speaker = self.df.drop_duplicates(subset="id_speaker").query(
            "id_speaker not in @self.banned_speakers"
        )
        while (
            (
                len(self) < self.min_samples
                or any(
                    count < len(self) * requirement
                    for key, requirement in self.requirements.items()
                    for count in self.counts[key].values()  # type: ignore[literal-required]
                )
            )
            and set(df_speaker.id_speaker.tolist()) - self.speakers != set()
            and len(self) < self.max_samples
        ):
            speakers = df_speaker["id_speaker"].tolist()
            scores = df_speaker.apply(func=self._give_score, axis=1).tolist()

            # Normalise the scores to probabilities
            probs = [score / sum(scores) for score in scores]

            # Ensure that the probabilities sum to 1, as this is required by the
            # `choice` function. We do this by changing the last probability to 1 - the
            # sum of the other probabilities. Sometimes, for some reason the other
            # probabilities sum to slightly more than 1, making the new probability
            # negative. In this case we clamp it to 0, and change the second last
            # probability to 1 - the sum of the other probabilities, and so on.
            index_to_change_if_sum_not_one = -1
            while sum(probs) != 1:
                sum_of_others = sum(probs) - probs[index_to_change_if_sum_not_one]
                probs[index_to_change_if_sum_not_one] = max(1 - sum_of_others, 0)
                index_to_change_if_sum_not_one -= 1

            speaker = self.rng.choice(speakers, p=probs)
            self.add_speaker_samples(speaker=speaker)

        if len(self) >= self.max_samples:
            self.satisfies_requirements = False

        return self

    def _update_weights(self) -> "EvalDataset":
        """Update the weights."""
        self.weights = {
            key: self._compute_weights(count=count, min_value=self.requirements[key])
            for key, count in self.counts.items()
        }
        return self

    def _compute_weights(self, count: dict[str, int], min_value: float) -> dict:
        """Compute weights for a feature, based on counts.

        The weights are computed as follows:
            1. We first normalise the counts to probabilities. This is because we want
               to sum up these weights for the different features, and using
               probabilities instead of counts ensures that they're on the same scale,
               making each feature equally important.
            2. Next, we divide the normalised counts by the minimum value required for
               the feature that we're currently computing the weights for. This will
               yield values that are 1 when the feature is at the minimum value.
            3. We next subtract these values from 1, which results in values that are 1
               when the feature is near zero, and less than or equal to zero when the
               feature is at the minimum required value.
            4. Since we don't want negative weights, we clamp the values to be at least
               a small positive value. We don't clamp to zero here, as we might dividing
               by zero later on in that case.

        Args:
            count:
                Counts for a feature.
            min_value:
                The minimally required value for the feature.

        Returns:
            Weight mapping for the feature.
        """
        normalised_counts = {
            key: 0 if len(self) == 0 else value / len(self)
            for key, value in count.items()
        }
        weights = {
            key: max(1 - value / min_value, 1e-6)
            for key, value in normalised_counts.items()
        }
        return weights

    def __repr__(self) -> str:
        """Return the string representation of the EvalDataset class."""
        num_hours = len(self) * self.mean_seconds_per_sample / 60 / 60
        msg = (
            f"\nEstimated number of hours: {num_hours:.2f}"
            f"\nDifficulty: {self.difficulty:.2f}"
            f"\nSpeaker IDs: {self.speakers}\n\n"
        )

        for key, count in self.counts.items():
            msg += f"{key.capitalize()} distribution:\n"
            dist = {
                feature: f"{feature_count / len(self):.0%}" if len(self) > 0 else "0%"
                for feature, feature_count in count.items()  # type: ignore[attr-defined]
            }
            for feature, feature_pct in dist.items():
                msg += f"- {feature}: {feature_pct}\n"

        return msg

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.indices)


def age_to_group(age: int, age_groups: list[AgeGroup]) -> str:
    """Return the age group of a given age.

    Args:
        age:
            The age of the speaker.
        age_groups:
            A list of the possible age groups.

    Returns:
        The age group of the speaker.

    Raises:
        ValueError:
            If the age is not in any age group.
    """
    for age_group in age_groups:
        if age in age_group:
            return str(age_group)
    raise ValueError(f"Age {age} not in any age group, out of {age_groups}.")


def load_coral_metadata_df(
    sub_dialect_to_dialect: dict[str, str],
    age_groups: list[tuple[int, int]],
    max_cer: float,
    streaming: bool,
    dataset_path: str,
    revision: str,
    cache_dir: str | None,
) -> pd.DataFrame:
    """Load the metadata of the CoRal dataset.

    If the metadata is not found, it will be downloaded.

    Args:
        sub_dialect_to_dialect:
            A mapping from sub-dialect to dialect.
        age_groups:
            The age groups to use for splitting.
        max_cer:
            The maximum CER of a sample.
        streaming:
            Whether to load the dataset in streaming mode. Only relevant if `dataset` is
            None.
        dataset_path:
            The path to the dataset to load.
        revision:
            The revision of the dataset to load. If None, the latest revision is used.
        cache_dir:
            The directory to cache the dataset in. If None then the standard cache
            directory is used.

    Returns:
        The metadata of the CoRal dataset.
    """
    metadata_path = Path("coral-metadata.csv")

    if metadata_path.exists():
        return pd.read_csv(metadata_path, low_memory=False)

    dataset = load_dataset(
        path=dataset_path,
        name="read_aloud",
        revision=revision,
        streaming=streaming,
        cache_dir=cache_dir,
    ).remove_columns("audio")

    if streaming:
        assert isinstance(dataset, IterableDatasetDict)

        # This will download the dataset with a progress bar, and remove the audio
        # column along the way, to save memory.
        num_samples = sum(
            split.info.splits[split_name].num_examples
            for split_name, split in dataset.items()
        )
        metadata = [
            sample
            for split in dataset.values()
            for sample in tqdm(
                split, total=num_samples, desc="Downloading CoRal dataset"
            )
        ]
        df = pd.DataFrame(metadata)

    else:
        assert isinstance(dataset, DatasetDict)
        merged_dataset = concatenate_datasets(
            dsets=[split for split in dataset.values() if split is not None]
        )
        df = pd.DataFrame(merged_dataset.to_pandas())

    logger.info(f"Downloaded CoRal metadata with {len(df):,} raw samples.")

    # Map the dialects to the dialect categories that we use for splitting
    all_dialects = set(df.dialect.unique())
    missing_dialects = all_dialects - set(sub_dialect_to_dialect.keys())
    if missing_dialects:
        raise ValueError(
            f"Missing dialects in sub_dialect_to_dialect mapping: {missing_dialects}"
        )
    df.dialect = df.dialect.map(sub_dialect_to_dialect)

    # For non-native speakers, we use the accent as the dialect
    df.country_birth = df.country_birth.map(lambda x: "DK" if x is None else x)
    df.loc[df.country_birth != "DK", "dialect"] = "Non-native"

    # We remove the nonbinary speakers from being in the validation and test sets,
    # since there are only 3 such speakers in the dataset - they will be part of the
    # training split instead.
    samples_before = len(df)
    df = df.query("gender != 'nonbinary'")
    samples_removed = samples_before - len(df)
    logger.info(f"Removed {samples_removed:,} nonbinary samples.")

    # Convert age to age group
    df["age_group"] = df.age.apply(
        lambda age: age_to_group(
            age=age,
            age_groups=[
                AgeGroup(min=min_age, max=max_age) for min_age, max_age in age_groups
            ],
        )
    )

    # Remove the manually rejected samples
    samples_before = len(df)
    df = df.query("validated != 'rejected' and validated != 'maybe'")
    samples_removed = samples_before - len(df)
    logger.info(f"Removed {samples_removed:,} manually rejected samples.")

    # Remove the automatically rejected samples
    samples_before = len(df)
    df = df.query("asr_cer < @max_cer")
    samples_removed = samples_before - len(df)
    logger.info(f"Removed {samples_removed:,} samples with CER > {max_cer}.")

    # Store the metadata for future use
    df.to_csv("coral-metadata.csv", index=False)

    return df


if __name__ == "__main__":
    main()
