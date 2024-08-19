"""Automatic validation of an ASR dataset.

Usage:
    python src/scripts/validate_coral_asr.py [key=value] [key=value] ...
"""

import logging
import re
import warnings
from time import sleep

import evaluate
import hydra
import torch
from coral.data import clean_example
from coral.data_models import Processor
from datasets import Audio, Dataset, DatasetDict, load_dataset
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

    logger.info("Validating the dataset...")
    new_data_dict: dict[str, Dataset] = dict()
    for split_name, split in processed_dataset.items():
        predictions, labels, wers = get_wers(dataset=split, transcriber=transcriber)
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
            .add_column(name="asr_wer", column=wers, new_fingerprint=split._fingerprint)
            .add_column(
                name="asr_validation_model",
                column=[config.model_id] * len(split),
                new_fingerprint=split._fingerprint,
            )
            .filter(lambda sample: sample["validated"] != "rejected")
        )
        if split_name in {"val", "test"}:
            new_split = new_split.filter(
                lambda x: x["asr_wer"] < config.max_val_test_wer
            )
        elif split_name == "train":
            new_split = new_split.filter(lambda x: x["asr_wer"] < config.max_train_wer)
        else:
            raise ValueError(f"Unknown split name: {split_name!r}")
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
    characters_to_keep: str,
    text_column: str,
    processor: Processor,
) -> DatasetDict:
    """Process a dataset for ASR.

    Args:
        dataset:
            The dataset to clean.
        characters_to_keep:
            The characters to keep in the transcriptions.
        text_column:
            The name of the column containing the transcriptions.
        processor:
            The processor used for processing the data.

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

    # This contains all the punctuation characters that will be removed from the
    # transcriptions, as they do not have an influence on the pronunciation of the
    # words.
    non_standard_characters_regex = re.compile(
        f"[^{re.escape(characters_to_keep + ' |')}]"
    )

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
        examples["labels"] = processor(
            text=examples[text_column], truncation=True
        ).input_ids
        return examples

    processed_dataset = dataset.map(process_examples, batched=True)

    return processed_dataset


def get_wers(
    dataset: Dataset, transcriber: AutomaticSpeechRecognitionPipeline
) -> tuple[list[str], list[float], list[float]]:
    """Get the word error rates for each sample in the dataset.

    Args:
        dataset:
            The dataset to validate.
        transcriber:
            The transcriber used for transcribing the audio.

    Returns:
        A triple (predictions, labels, wers) where:
            predictions:
                The transcriptions predicted by the model.
            labels:
                The ASR-processed ground-truth labels for each sample.
            wers:
                The word error rates for each sample.
    """
    predictions: list[str] = list()
    key_dataset = KeyDataset(dataset=dataset, key="audio")
    for out in tqdm(transcriber(inputs=key_dataset), desc="Transcribing"):
        prediction = out["text"].strip()
        predictions.append(prediction)

    labels = dataset["text"]

    # Compute the word error rates
    wer_metric = evaluate.load("wer")
    wers = [
        wer_metric.compute(predictions=[pred], references=[ref])
        for pred, ref in zip(tqdm(predictions, desc="Computing WERs"), labels)
    ]

    # Ensure that the WERs are indeed floats, as `compute` returns a dictionary for some
    # metrics
    wers = [wer if isinstance(wer, float) else -100.0 for wer in wers]
    assert all(wer >= 0 for wer in wers), (
        "The number of WERs should be equal to the number of predictions - found "
        f"{len(wers):,} WERs and {len(predictions):,} predictions."
    )

    return predictions, labels, wers


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
