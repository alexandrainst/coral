"""Evaluation of Wav2Vec 2.0 models on ASR datasets."""

from functools import partial

import hydra
import transformers.utils.logging as tf_logging
from datasets import Audio, Dataset, IterableDataset
from omegaconf import DictConfig
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
)

from coral_models.asr.wav2vec2.clean import clean_dataset
from coral_models.asr.wav2vec2.compute_metrics import compute_metrics
from coral_models.asr.wav2vec2.data_collator import DataCollatorCTCWithPadding
from coral_models.data import load_data


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Evaluate ASR models on a dataset"""
    dataset: Dataset | IterableDataset = load_data(cfg)["test"]

    # Load the pretrained processor and model
    if cfg.model.language_model_decoder is None:
        processor = Wav2Vec2Processor.from_pretrained(cfg.hub_id, use_auth_token=True)
    else:
        try:
            processor = Wav2Vec2ProcessorWithLM.from_pretrained(
                cfg.hub_id, use_auth_token=True
            )
        except (FileNotFoundError, ValueError):
            processor = Wav2Vec2Processor.from_pretrained(
                cfg.hub_id, use_auth_token=True
            )
    model = Wav2Vec2ForCTC.from_pretrained(cfg.hub_id, use_auth_token=True)

    # Clean and tokenize the transcriptions
    dataset = clean_dataset(cfg, dataset=dataset)
    dataset = dataset.cast_column(
        column="audio", feature=Audio(sampling_rate=cfg.dataset.sampling_rate)
    )
    dataset = preprocess_transcriptions(
        dataset=dataset, processor=processor, text_column=cfg.dataset.text_column
    )

    # Initialise data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

    # Initialise the trainer
    trainer = Trainer(
        args=TrainingArguments(".", remove_unused_columns=False),
        model=model,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, processor=processor),
        eval_dataset=dataset,
        tokenizer=processor.tokenizer,
    )

    # Disable most of the `transformers` logging
    tf_logging.set_verbosity_error()

    # Remove trainer logging
    trainer.log = lambda _: None

    # Evaluate the model
    metrics = trainer.evaluate(dataset)

    # Extract the WER
    wer = 100 * metrics["eval_wer"]

    # Print the metrics
    print(f"\n*** RESULTS ON {cfg.dataset.name} ***")
    print(f"{cfg.hub_id} achieved a WER of {wer:.2f}.\n")


def preprocess_transcriptions(
    dataset: Dataset | IterableDataset,
    processor: Wav2Vec2Processor | Wav2Vec2ProcessorWithLM,
    text_column: str = "sentence",
) -> Dataset | IterableDataset:
    def tokenize_examples(example: dict) -> dict:
        example["labels"] = processor(
            text=example[text_column],
            truncation=True,
        ).input_ids
        example["input_length"] = len(example["labels"])
        return example

    mapped = dataset.map(tokenize_examples)

    # After calling `map` the DatasetInfo is lost, so we need to add it back in
    mapped._info = dataset._info
    return mapped


if __name__ == "__main__":
    main()
