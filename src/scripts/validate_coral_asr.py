"""Automatic validation of an ASR dataset.

Usage:
    python src/scripts/validate_coral_asr.py [key=value] [key=value] ...
"""

import logging
import multiprocessing as mp
import re
import warnings
from time import sleep

import evaluate
import hydra
import torch
from coral.data import clean_example
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    enable_progress_bar,
    load_dataset,
)
from omegaconf import DictConfig
from requests import HTTPError
from tqdm.auto import tqdm
from transformers import AutomaticSpeechRecognitionPipeline, pipeline
from transformers.pipelines.pt_utils import KeyDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_coral_asr")

warnings.filterwarnings("ignore", category=FutureWarning)


@hydra.main(
    config_path="../../config", config_name="dataset_validation", version_base=None
)
def main(config: DictConfig) -> None:
    """Validate the samples of the CoRal ASR dataset using an ASR model.

    Args:
        config:
            The Hydra configuration object.
    """
    enable_progress_bar()

    logger.info(f"Loading the {config.dataset_id!r} dataset...")
    dataset = load_dataset(
        path=config.dataset_id,
        name=config.dataset_subset,
        revision=config.dataset_revision,
        token=True,
        cache_dir=config.cache_dir,
        streaming=True,
    )
    if isinstance(dataset, IterableDataset):
        dataset = IterableDatasetDict(dict(train=dataset))
    assert isinstance(dataset, IterableDatasetDict)

    # num_samples_before = sum(len(split) for split in dataset.values())
    dataset = dataset.filter(
        lambda samples: [
            audio_dct["array"].shape[0]
            > audio_dct["sampling_rate"] * config.min_seconds_per_example
            for audio_dct in samples[config.audio_column]
        ],
        batched=True,
        # num_proc=mp.cpu_count(),
        # desc="Filtering out samples with too short audio",
    )
    # num_short_samples_removed = num_samples_before - sum(
    #     len(split) for split in dataset.values()
    # )
    dataset = dataset.filter(
        lambda samples: [
            audio_dct["array"].shape[0]
            < audio_dct["sampling_rate"] * config.max_seconds_per_example
            for audio_dct in samples[config.audio_column]
        ],
        batched=True,
        # num_proc=mp.cpu_count(),
        # desc="Filtering out samples with too long audio",
    )
    # num_long_samples_removed = (
    #     num_samples_before
    #     - num_short_samples_removed
    #     - sum(len(split) for split in dataset.values())
    # )
    # logger.info(
    #     f"Filtered out {num_short_samples_removed:,} samples with too short audio "
    #     f"(< {config.min_seconds_per_example} seconds) and {num_long_samples_removed:,} "
    #     f"samples with too long audio (> {config.max_seconds_per_example} seconds)."
    # )

    for split_name, split in dataset.items():
        # num_samples_before = len(split)
        if split_name == config.train_split:
            dataset[split_name] = split.filter(
                lambda samples: [
                    validated != "rejected" for validated in samples["validated"]
                ],
                batched=True,
                # num_proc=mp.cpu_count(),
            )
            # msg = (
            #     "Filtered out {num_samples_removed:,} samples with a 'rejected' "
            #     f"validation status from the {split_name} split."
            # )
        else:
            dataset[split_name] = split.filter(
                lambda samples: [
                    validated != "rejected" and validated != "maybe"
                    for validated in samples["validated"]
                ],
                batched=True,
                # num_proc=mp.cpu_count(),
            )
            # msg = (
            #     "Filtered out {num_samples_removed:,} samples with a 'rejected' or "
            #     f"'maybe' validation status from the {split_name} split."
            # )
        # num_samples_removed = num_samples_before - len(dataset[split_name])
        # logger.info(msg.format(num_samples_removed=num_samples_removed))

    # This contains all the punctuation characters that will be removed from the
    # transcriptions, as they do not have an influence on the pronunciation of the
    # words.
    non_standard_characters_regex = re.compile(
        f"[^{re.escape(config.characters_to_keep + ' |')}]"
    )

    logger.info("Processing the dataset...")
    processed_dataset = process_dataset(
        dataset=dataset,
        non_standard_characters_regex=non_standard_characters_regex,
        text_column=config.text_column,
        audio_column=config.audio_column,
        sample_rate=config.sample_rate,
    )

    logger.info(f"Loading the {config.model_id!r} ASR model...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    transcriber = pipeline(
        task="automatic-speech-recognition", model=config.model_id, device=device
    )
    assert isinstance(transcriber, AutomaticSpeechRecognitionPipeline)

    new_data_dict: dict[str, Dataset] = dict()
    metric_names = [metric.name.lower() for metric in config.metrics]
    for split_name, split in processed_dataset.items():
        logger.info(f"Validating the {split_name} split of the dataset...")
        predictions, labels, score_dict = compute_metrics(
            dataset=split,
            transcriber=transcriber,
            metric_names=metric_names,
            non_standard_characters_regex=non_standard_characters_regex,
            text_column=config.text_column,
            batch_size=config.batch_size,
        )

        # Create a new split with the predictions, labels, and scores
        new_split = (
            dataset[split_name]
            .add_column(
                name="asr_prediction",
                column=predictions,
                new_fingerprint=split._fingerprint,
            )
            .add_column(
                name="asr_label", column=labels, new_fingerprint=split._fingerprint
            )
            .add_column(
                name="asr_validation_model",
                column=[config.model_id] * len(split),
                new_fingerprint=split._fingerprint,
            )
        )
        for metric_name, scores in score_dict.items():
            new_split = new_split.add_column(
                name=f"asr_{metric_name.lower()}",
                column=scores,
                new_fingerprint=split._fingerprint,
            )
        new_data_dict[split_name] = new_split

    # Filter the dataset based on the metrics from the validation model
    new_dataset = DatasetDict(new_data_dict)
    metrics_with_constraints = [
        metric for metric in config.metrics if "max" in metric or "min" in metric
    ]
    for metric in metrics_with_constraints:
        num_samples_before = sum(len(split) for split in new_dataset.values())
        if "max" in metric:
            new_dataset = new_dataset.filter(
                lambda samples: [
                    score < metric.max
                    for score in samples[f"asr_{metric.name.lower()}"]
                ],
                batched=True,
                num_proc=mp.cpu_count(),
            )
            msg = (
                f"Filtered out {{num_samples_removed:,}} samples with a {metric.name} "
                f"score greater than {metric.max:.2f}."
            )
        else:
            new_dataset = new_dataset.filter(
                lambda samples: [
                    score > metric.min
                    for score in samples[f"asr_{metric.name.lower()}"]
                ],
                batched=True,
                num_proc=mp.cpu_count(),
            )
            msg = (
                f"Filtered out {{num_samples_removed:,}} samples with a {metric.name} "
                f"score lower than {metric.min:.2f}."
            )
        num_samples_removed = num_samples_before - sum(
            len(split) for split in new_dataset.values()
        )
        logger.info(msg.format(num_samples_removed=num_samples_removed))

    logger.info(f"Uploading the validated dataset to {config.output_dataset_id!r}...")
    for _ in range(60):
        try:
            new_dataset.push_to_hub(
                repo_id=config.output_dataset_id,
                config_name=config.output_dataset_subset,
                max_shard_size="500MB",
                commit_message="Add ASR validation",
                private=True,
            )
            logger.info("All done!")
            break
        except (RuntimeError, HTTPError) as e:
            logger.info(f"Error while pushing to hub: {e}")
            logger.info("Waiting a minute before trying again...")
            sleep(60)
            logger.info("Retrying...")
    else:
        logger.error("Failed to upload the dataset to the Hugging Face Hub.")


def process_dataset(
    dataset: DatasetDict,
    non_standard_characters_regex: re.Pattern[str],
    text_column: str,
    audio_column: str,
    sample_rate: int,
) -> DatasetDict:
    """Process a dataset for ASR.

    Args:
        dataset:
            The dataset to clean.
        non_standard_characters_regex:
            Regular expression that matches all characters that should be removed from
            the transcriptions.
        text_column:
            The name of the column containing the transcriptions.
        audio_column:
            The name of the column containing the audio.
        max_seconds_per_example:
            The maximum number of seconds that an example can have.
        sample_rate:
            The desired sampling rate of the audio.

    Returns:
        The processed dataset.
    """
    # logger.info("Casting the audio to the correct sampling rate...")
    # processed_dataset = dataset.cast_column( column=audio_column, feature=Audio(sampling_rate=sample_rate))

    # Dictionary that contains characters to be converted (from the key to the value).
    # Some values contain spaces to ensure that they're separated from other
    # characters, and superfluous spaces are removed later. Note also that these are
    # converted in the order they appear in the dictionary.
    conversion_dict = {
        "aa": "å",
        "ğ": "g",
        "ñ": "n",
        "ń": "n",
        "è": "e",
        "kg": " kilo ",
        "μg": " mikrogram ",
        "-": " minus ",
        "+": " plus ",
        "μ": " mikro ",
        "§": " paragraf ",
        "%": " procent ",
        "‰": " promille ",
        "ú": "u",
        "ş": "s",
        "ê": "e",
        "ã": "a",
        "ë": "e",
        "ć": "c",
        "ä": "æ",
        "í": "i",
        "š": "s",
        "î": "i",
        "ě": "e",
        "ð": "d",
        "á": "a",
        "ó": "o",
        "þ": "th",
        "ı": "i",
        "ö": "ø",
        "ç": "c",
        "ș": "s",
        "\u0301": " ",  # Empty whitespace symbol
        "\u200b": " ",  # Empty whitespace symbol
    }

    def clean_examples(examples: dict[str, list]) -> dict:
        """Clean the transcriptions in the examples.

        Args:
            examples:
                The examples to clean.

        Returns:
            The cleaned examples.
        """
        examples[text_column] = [
            clean_example(
                example={text_column: sample_text},
                non_standard_characters_regex=non_standard_characters_regex,
                conversion_dict=conversion_dict,
                text_column=text_column,
            )[text_column]
            for sample_text in examples[text_column]
        ]
        return examples

    processed_dataset = dataset.map(
        clean_examples,
        batched=True,  # , desc="Cleaning dataset", num_proc=mp.cpu_count()
    )

    return processed_dataset


def compute_metrics(
    dataset: Dataset,
    transcriber: AutomaticSpeechRecognitionPipeline,
    metric_names: list[str],
    non_standard_characters_regex: re.Pattern[str],
    text_column: str,
    batch_size: int,
) -> tuple[list[str], list[str], dict[str, list[float]]]:
    """Compute the metrics for the dataset.

    Args:
        dataset:
            The dataset to validate.
        transcriber:
            The transcriber used for transcribing the audio.
        metric_names:
            The names of the metrics to compute. Needs to be compatible with the name of
            the metric in the `evaluate` library.
        non_standard_characters_regex:
            Regular expression that matches all characters that should be removed from
            the transcriptions.
        text_column:
            The name of the column containing the transcriptions.
        batch_size:
            The batch size to use for transcribing the audio.

    Returns:
        A triple (predictions, labels, cers, wers) where:
            predictions:
                The transcriptions predicted by the model.
            labels:
                The ASR-processed ground-truth labels for each sample.
            cers:
                The word error rates for each sample.
            wers:
                The word error rates for each sample.
    """
    predictions: list[str] = list()
    # labels = [lbl.lower().strip() for lbl in dataset[text_column]]
    labels: list[str] = []

    with tqdm(total=len(dataset), desc="Transcribing") as pbar:
        for out in transcriber(KeyDataset(dataset, "audio"), batch_size=batch_size):
            prediction = re.sub(
                pattern=non_standard_characters_regex,
                repl="",
                string=out["text"].strip().lower(),
            )
            breakpoint()
            predictions.append(prediction.strip())
            pbar.update()

    all_scores: dict[str, list[float]] = dict()
    for metric_name in metric_names:
        metric = evaluate.load(metric_name)
        scores = [
            metric.compute(predictions=[pred], references=[ref])
            for pred, ref in zip(
                tqdm(predictions, desc=f"Computing {metric_name.upper()}s"), labels
            )
        ]

        # Ensure that the scores are indeed floats, as `compute` returns a dictionary for
        # some metrics
        scores = [score if isinstance(score, float) else -100.0 for score in scores]
        assert all(score >= 0 for score in scores), (
            f"The number of {metric_name.upper()}s should be equal to the number "
            f"of predictions - found {len(scores):,} {metric_name.upper()}s and "
            f"{len(predictions):,} predictions."
        )

        all_scores[metric_name] = scores

    return predictions, labels, all_scores


def preprocess_logits_for_metrics(
    logits: torch.Tensor, _: torch.Tensor
) -> torch.Tensor:
    """Workaround to avoid storing too many tensors that are not needed.

    Args:
        logits:
            The logits from the model, of shape (batch_size, seq_len, num_labels).
        labels:
            The labels for the logits - not used here.

    Returns:
        The prediction token IDs to use for computing the metrics.
    """
    assert isinstance(logits, torch.Tensor)
    assert logits.ndim == 3
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


if __name__ == "__main__":
    main()
