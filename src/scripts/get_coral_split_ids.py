"""Get the speaker IDs for the CoRal test and validation splits.

The test set is subject to the following constraints:
    - At least 7.5 hours
    - At least 40% of each gender
    - At least 20% of each age group (0-24, 25-49, 50+)
    - At least 10% of each dialect group
    - At least 5% of speakers with a foreign accent

The validation split has no formal criteria, but must be significantly smaller than the
test set and should have roughly the same distribution.

Developers:
    - Oliver Kinch (oliver.kinch@alexandra.dk)
    - Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
"""

import logging
from collections import namedtuple
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from datasets import IterableDataset, load_dataset
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("get_coral_split_ids")


# Constants related to minimum requirements
MEAN_SECONDS_PER_SAMPLE = 5
MIN_TEST_HOURS = 7.5
MAX_TEST_HOURS = 20.0
MIN_VAL_HOURS = 1.0
MAX_VAL_HOURS = 10.0

GENDERS = ["female", "male"]
DIALECTS = [
    "Bornholmsk",
    "Fynsk",
    "Københavnsk",
    "Nordjysk",
    "Sjællandsk",
    "Sydømål",
    "Sønderjysk",
    "Vestjysk",
    "Østjysk",
]

age_group = namedtuple("age_group", ["min", "max"])
AGE_GROUPS = {
    "0-24": age_group(0, 25),
    "25-49": age_group(25, 50),
    "50-": age_group(50, int(1e6)),
}
ACCENTS = ["native", "foreign"]

SUB_DIALECT_TO_DIALECT = {
    "midtøstjysk": "Østjysk",
    "østjysk": "Østjysk",
    "amagermål": "Københavnsk",
    "nørrejysk": "Nordjysk",
    "vestjysk": "Vestjysk",
    "nordsjællandsk": "Sjællandsk",
    "sjællandsk": "Sjællandsk",
    "fynsk": "Fynsk",
    "bornholmsk": "Bornholmsk",
    "sønderjysk": "Sønderjysk",
    "vendsysselsk (m. hanherred og læsø)": "Nordjysk",
    "østligt sønderjysk (m. als)": "Sønderjysk",
    "nordvestsjællandsk": "Sjællandsk",
    "thybomål": "Vestjysk",
    "himmerlandsk": "Nordjysk",
    "djurslandsk (nord-, syddjurs m. nord- og sydsamsø, anholt)": "Østjysk",
    "sydsjællandsk (sydligt sydsjællandsk)": "Sjællandsk",
    "sydfynsk": "Fynsk",
    "morsingmål": "Vestjysk",
    "sydøstjysk": "Østjysk",
    "østsjællandsk": "Sjællandsk",
    "syd for rigsgrænsen: mellemslesvisk, angelmål, fjoldemål": "Sønderjysk",
    "vestfynsk (nordvest-, sydvestfynsk)": "Fynsk",
    "vestlig sønderjysk (m. mandø og rømø)": "Sønderjysk",
    "sydvestjysk (m. fanø)": "Vestjysk",
    "sallingmål": "Vestjysk",
    "nordfalstersk": "Sydømål",
    "langelandsk": "Fynsk",
    "sydvestsjællandsk": "Sjællandsk",
    "lollandsk": "Sydømål",
    "sydømål": "Sydømål",
    "ommersysselsk": "Østjysk",
    "sydfalstersk": "Sydømål",
    "fjandbomål": "Vestjysk",
}


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
        seed (int):
            The seed for the random number generator.
        indices (list[int]):
            List of indices of the Coral dataset that will be included in the dataset.
        speakers (set[str]):
            List of speaker IDs that will be included in the dataset.
        rng (np.random.Generator):
            Random number generator.
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
        """
        self.df = df
        self.min_samples: int = min_samples
        self.max_samples: int = max_samples
        self.requirements: dict[str, float] = requirements
        self.banned_speakers: set[str] = banned_speakers
        self.seed: int = seed
        self.indices: list[int] = list()
        self.speakers: set[str] = set()
        self.rng = np.random.default_rng(seed=seed)

        self.counts = dict(
            gender={gender: 0 for gender in GENDERS},
            dialect={dialect: 0 for dialect in DIALECTS},
            age_group={age_group: 0 for age_group in AGE_GROUPS.keys()},
            accent={accent: 0 for accent in ACCENTS},
        )

        self.weights = {
            key: self._make_weights(count=count, beta=0)
            for key, count in self.counts.items()
        }

        self.betas = dict(dialect=100.0, age_group=5.0)

    def add_speaker_samples(self, speaker: str) -> None:
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
                row["age"] = age_to_group(age=row["age"])
            count[row[key]] += n_samples

        self._update_weights()

    def add_dialect_samples(self):
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
                    for count in self.counts[key].values()
                )
            )
            and set(df_speaker.id_speaker.tolist()) - self.speakers != set()
            and len(self) < self.max_samples
        ):

            def _give_score(row: pd.Series) -> float:
                return sum(weight[row[key]] for key, weight in self.weights.items())

            speakers = df_speaker["id_speaker"].tolist()
            scores = df_speaker.apply(func=_give_score, axis=1).tolist()
            probs = (
                torch.softmax(torch.tensor(scores), dim=0)
                .clamp(min=torch.tensor(0), max=torch.tensor(1))
                .tolist()
            )

            index_to_change_if_sum_not_one = -1
            while sum(probs) != 1:
                sum_of_others = sum(probs) - probs[index_to_change_if_sum_not_one]
                probs[index_to_change_if_sum_not_one] = max(1 - sum_of_others, 0)
                index_to_change_if_sum_not_one -= 1

            speaker = self.rng.choice(speakers, p=probs)
            self.add_speaker_samples(speaker=speaker)

        return self

    def _update_weights(self) -> None:
        """Update the weights."""
        self.weights = {
            key: self._make_weights(count=count, beta=self.betas.get(key, 0))
            for key, count in self.counts.items()
        }

    def _make_weights(self, count: dict, beta: float) -> dict:
        """Make weights based on counts.

        Args:
            count:
                Counts for a feature.
            beta:
                Shift the weights of the least represented feature.

        Returns:
            Weights for the feature.
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
        num_hours = len(self) * MEAN_SECONDS_PER_SAMPLE / 60 / 60
        msg = (
            f"\nEstimated number of hours: {num_hours:.2f}"
            f"\nSpeaker IDs: {self.speakers}\n\n"
        )

        for key, count in self.counts.items():
            msg += f"{key.capitalize()} distribution:\n"
            dist = {
                feature: f"{feature_count / len(self):.0%}" if len(self) > 0 else "0%"
                for feature, feature_count in count.items()
            }
            for feature, feature_pct in dist.items():
                msg += f"- {feature}: {feature_pct}\n"

        return msg

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.indices)


def age_to_group(age: int) -> str:
    """Return the age group of a given age.

    Args:
        age:
            The age of the speaker.

    Returns:
        The age group of the speaker.

    Raises:
        ValueError:
            If the age is not in any age group.
    """
    for group, age_group in AGE_GROUPS.items():
        if age_group.min <= age < age_group.max:
            return group
    raise ValueError(f"Age {age} not in any age group.")


def load_coral_metadata_df() -> pd.DataFrame:
    """Load the metadata of the CoRal dataset.

    If the metadata is not found, it will be downloaded.

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
        df.dialect = df.dialect.map(SUB_DIALECT_TO_DIALECT)
        df["accent"] = df.language_native.apply(
            lambda x: "native" if x == "da" else "foreign"
        )
        df = df.query("gender != 'nonbinary'")
        df.to_csv("coral-metadata.csv", index=False)

    return df


@click.command()
@click.option(
    "--num-attempts",
    "-n",
    default=1000,
    help="Number of attempts to find the best test and validation splits.",
)
def main(num_attempts: int) -> None:
    """Main function to get the speaker IDs for the CoRal test and validation splits."""
    df = load_coral_metadata_df()

    # Build test split
    test_requirements = dict(gender=0.4, age_group=0.2, dialect=0.1, accent=0.05)
    test_datasets: list[Dataset] = list()
    for seed in tqdm(range(4242, 4242 + num_attempts), desc="Computing test splits"):
        test_dataset = Dataset(
            df=df,
            min_samples=int(MIN_TEST_HOURS * 60 * 60 / MEAN_SECONDS_PER_SAMPLE),
            max_samples=int(MAX_TEST_HOURS * 60 * 60 / MEAN_SECONDS_PER_SAMPLE),
            requirements=test_requirements,
            banned_speakers=set(),
            seed=seed,
        ).add_dialect_samples()
        test_datasets.append(test_dataset)
    test_dataset = min(test_datasets, key=lambda x: len(x))
    logger.info(f"Test dataset:\n{test_dataset}")

    # Build validation split
    val_requirements = dict(gender=0.2, age_group=0.1, dialect=0.01, accent=0.01)
    val_datasets: list[Dataset] = list()
    for seed in range(4242, 4242 + num_attempts):
        val_dataset = Dataset(
            df=df,
            min_samples=int(MIN_VAL_HOURS * 60 * 60 / MEAN_SECONDS_PER_SAMPLE),
            max_samples=int(MAX_VAL_HOURS * 60 * 60 / MEAN_SECONDS_PER_SAMPLE),
            requirements=val_requirements,
            banned_speakers=test_dataset.speakers,
            seed=seed,
        ).add_dialect_samples()
        val_datasets.append(val_dataset)
    val_dataset = min(val_datasets, key=lambda x: len(x))
    logger.info(f"Validation dataset:\n{val_dataset}")

    assert set(test_dataset.speakers).intersection(val_dataset.speakers) == set()


if __name__ == "__main__":
    main()
