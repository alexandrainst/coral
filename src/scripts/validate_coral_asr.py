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
from datasets import Audio, Dataset, DatasetDict, enable_progress_bar, load_dataset
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
    logger.info(f"Loading the {config.dataset_id!r} dataset...")
    dataset = load_dataset(
        path=config.dataset_id,
        name=config.dataset_subset,
        split=config.dataset_split,
        revision=config.dataset_revision,
        token=True,
        cache_dir=config.cache_dir,
    )
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({config.dataset_split: dataset})
    assert isinstance(dataset, DatasetDict)

    logger.info("Resampling audio to 16kHz...")
    processed_dataset = dataset.cast_column(
        column=config.audio_column, feature=Audio(sampling_rate=16_000)
    )

    # This contains all the punctuation characters that will be removed from the
    # transcriptions, as they do not have an influence on the pronunciation of the
    # words.
    non_standard_characters_regex = re.compile(
        f"[^{re.escape(config.characters_to_keep + ' |')}]"
    )

    logger.info("Processing the dataset...")
    processed_dataset = process_dataset(
        dataset=processed_dataset,
        non_standard_characters_regex=non_standard_characters_regex,
        text_column=config.text_column,
    )
    assert isinstance(processed_dataset, DatasetDict)

    logger.info(f"Loading the {config.model_id!r} ASR model...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=config.model_id,
        device=device,
        batch_size=config.batch_size,
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
        )
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
        new_split = new_split.filter(lambda sample: sample["validated"] != "rejected")
        for metric in config.metrics:
            if "max" in metric:
                new_split = new_split.filter(
                    lambda sample: sample[f"asr_{metric.name.lower()}"] < metric.max
                )
            elif "min" in metric:
                new_split = new_split.filter(
                    lambda sample: sample[f"asr_{metric.name.lower()}"] > metric.min
                )
        new_data_dict[split_name] = new_split

    logger.info(f"Uploading the validated dataset to {config.output_dataset_id!r}...")
    new_dataset = DatasetDict(new_data_dict)
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

    Returns:
        The processed dataset.
    """
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

    def process_examples(examples: dict[str, list]) -> dict:
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

    enable_progress_bar()
    processed_dataset = dataset.map(
        process_examples,
        batched=True,
        desc="Processing dataset",
        num_proc=mp.cpu_count(),
        batch_size=10_000,
    )
    return processed_dataset


def compute_metrics(
    dataset: Dataset,
    transcriber: AutomaticSpeechRecognitionPipeline,
    metric_names: list[str],
    non_standard_characters_regex: re.Pattern[str],
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
    key_dataset = KeyDataset(dataset=dataset, key="audio")
    with tqdm(total=len(dataset), desc="Transcribing") as pbar:
        for out in transcriber(key_dataset):
            assert isinstance(out, dict) and isinstance(out.get("text"), str)
            prediction = re.sub(
                pattern=non_standard_characters_regex,
                repl="",
                string=out["text"].strip().lower(),
            )
            predictions.append(prediction.strip())
            pbar.update()

    labels = [lbl.lower().strip() for lbl in dataset["text"]]

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
