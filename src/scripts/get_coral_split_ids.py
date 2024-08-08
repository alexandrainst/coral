"""Get the speaker IDs for the CoRal test and validation splits.

The test set is subject to the following constraints:
    - At least 7.5 hours
    - At least 40% of each gender
    - At least 20% of each age group (0-24, 25-49, 50+)
    - At least 10% of each dialect group
    - At least 5% of speakers with a foreign accent

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
from pathlib import Path
from typing import NamedTuple

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import IterableDataset, load_dataset
from omegaconf import DictConfig
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("get_coral_split_ids")


class AgeGroup(NamedTuple):
    """Named tuple to represent an age group."""

    min: int
    max: int | None

    def __repr__(self) -> str:
        """Return the string representation of the AgeGroup class."""
        if self.max is None:
            return f"{self.min}-"
        return f"{self.min}-{self.max - 1}"

    def __in__(self, age: int) -> bool:
        """Check if an age is in the age group.

        Args:
            age:
                The age to check.

        Returns:
            Whether the age is in the age group.
        """
        return self.min <= age and (self.max is None or age < self.max)


class Dataset:
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
        betas (dict):
            Shift the weights of the least represented feature.
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
        accents: list[str],
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
            accents:
                A list of possible accent values.
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
            accent={accent: 0 for accent in accents},
        )
        self.weights: dict[str, dict[str, float]] = {
            key: self._make_weights(count, beta=0) for key, count in self.counts.items()
        }
        self.betas = dict(dialect=100.0, age_group=5.0)
        self.add_dialect_samples()

    def add_speaker_samples(self, speaker: str) -> "Dataset":
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
            if key == "age":
                row["age"] = age_to_group(age=row["age"], age_groups=self.age_groups)
            count[row[key]] += n_samples

        self._update_weights()

        return self

    def add_dialect_samples(self) -> "Dataset":
        """Get samples of dialects each dialect.

        Args:
            dataset:
                Dataset object.

        Returns:
            Dataset object with samples of each dialect.
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

            def _give_score(row: pd.Series) -> float:
                return sum(
                    weight[row[key]]  # type: ignore[index]
                    for key, weight in self.weights.items()
                )

            speakers = df_speaker["id_speaker"].tolist()
            scores = df_speaker.apply(func=_give_score, axis=1).tolist()
            probs = (
                torch.softmax(torch.tensor(scores), dim=0)
                .clamp(min=torch.tensor(0), max=torch.tensor(1))
                .tolist()
            )

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

        return self

    def _update_weights(self) -> "Dataset":
        """Update the weights."""
        self.weights = {
            key: self._make_weights(count, beta=self.betas.get(key, 0))
            for key, count in self.counts.items()
        }
        return self

    def _make_weights(self, count: dict[str, int], beta: float) -> dict:
        """Make a weight mapping for a feature, based on counts.

        Args:
            count:
                Counts for a feature.
            beta:
                Shift the weights of the least represented feature.

        Returns:
            Weight mapping for the feature.
        """
        inv_count = {key: 1 / (1 + value) for key, value in count.items()}
        normalizer = sum(inv_count.values())
        weights = {key: value / normalizer for key, value in inv_count.items()}

        # Increase chance of sampling the least represented feature
        max_key = max(weights, key=weights.get)  # type: ignore[arg-type]
        weights[max_key] += weights[max_key] * beta

        return weights

    def __repr__(self) -> str:
        """Return the string representation of the Dataset class."""
        num_hours = len(self) * self.mean_seconds_per_sample / 60 / 60
        msg = (
            f"\nEstimated number of hours: {num_hours:.2f}"
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
    raise ValueError(f"Age {age} not in any age group.")


def load_coral_metadata_df(sub_dialect_to_dialect: dict[str, str]) -> pd.DataFrame:
    """Load the metadata of the CoRal dataset.

    If the metadata is not found, it will be downloaded.

    Args:
        sub_dialect_to_dialect:
            A mapping from sub-dialect to dialect.

    Returns:
        The metadata of the CoRal dataset.
    """
    metadata_path = Path("coral-metadata.csv")

    if metadata_path.exists():
        df = pd.read_csv(metadata_path, low_memory=False)
    else:
        coral = load_dataset(
            path="alexandrainst/coral", split="train", streaming=True
        ).remove_columns("audio")
        assert isinstance(coral, IterableDataset)

        coral_splits: dict | None = coral.info.splits
        assert coral_splits is not None, "No splits found in CoRal dataset."

        metadata = [
            sample
            for sample in tqdm(
                coral,
                total=coral_splits["train"].num_examples,
                desc="Downloading CoRal dataset",
            )
        ]
        df = pd.DataFrame(metadata)
        df.dialect = df.dialect.map(sub_dialect_to_dialect)
        df["accent"] = df.language_native.apply(
            lambda x: "native" if x == "da" else "foreign"
        )
        df = df.query("gender != 'nonbinary'")
        df.to_csv("coral-metadata.csv", index=False)

    return df


@hydra.main(
    config_path="../../config", config_name="dataset_creation", version_base=None
)
def main(config: DictConfig) -> None:
    """Main function to get the speaker IDs for the CoRal test and validation splits.

    Args:
        config:
            The Hydra configuration object
    """
    mean_seconds_per_sample = config.mean_seconds_per_sample
    num_attempts = config.num_split_attempts
    df = load_coral_metadata_df(sub_dialect_to_dialect=config.sub_dialect_to_dialect)

    # Build test split
    test_candidates: list[Dataset] = list()
    min_test_hours = config.requirements.test.min_hours
    max_test_hours = config.requirements.test.max_hours
    for seed in tqdm(range(4242, 4242 + num_attempts), desc="Computing test splits"):
        test_candidate = Dataset(
            df=df,
            min_samples=int(min_test_hours * 60 * 60 / mean_seconds_per_sample),
            max_samples=int(max_test_hours * 60 * 60 / mean_seconds_per_sample),
            requirements=dict(
                gender=config.requirements.test.gender_pct,
                dialect=config.requirements.test.dialect_pct,
                age_group=config.requirements.test.age_group_pct,
                accent=config.requirements.test.accent_pct,
            ),
            banned_speakers=set(),
            seed=seed,
            genders=config.genders,
            dialects=config.dialects,
            age_groups=config.age_groups,
            accents=config.accents,
            mean_seconds_per_sample=mean_seconds_per_sample,
        )
        test_candidates.append(test_candidate)
    test_dataset = min(test_candidates, key=lambda x: len(x))
    logger.info(f"Test dataset:\n{test_dataset}")

    # Build validation split
    val_candidates: list[Dataset] = list()
    min_val_hours = config.requirements.val.min_hours
    max_val_hours = config.requirements.val.max_hours
    for seed in tqdm(range(4242, 4242 + num_attempts), desc="Computing val splits"):
        val_candidate = Dataset(
            df=df,
            min_samples=int(min_val_hours * 60 * 60 / mean_seconds_per_sample),
            max_samples=int(max_val_hours * 60 * 60 / mean_seconds_per_sample),
            requirements=dict(
                gender=config.requirements.val.gender_pct,
                dialect=config.requirements.val.dialect_pct,
                age_group=config.requirements.val.age_group_pct,
                accent=config.requirements.val.accent_pct,
            ),
            banned_speakers=test_dataset.speakers,
            seed=seed,
            genders=config.genders,
            dialects=config.dialects,
            age_groups=config.age_groups,
            accents=config.accents,
            mean_seconds_per_sample=mean_seconds_per_sample,
        )
        val_candidates.append(val_candidate)
    val_dataset = min(val_candidates, key=lambda x: len(x))
    logger.info(f"Validation dataset:\n{val_dataset}")

    assert set(test_dataset.speakers).intersection(val_dataset.speakers) == set()


if __name__ == "__main__":
    main()
