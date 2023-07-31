"""Evaluation of Wav2Vec 2.0 models on ASR datasets."""

import hydra
from datasets import Audio, DatasetDict, IterableDatasetDict, Sequence, Value
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

from coral_models.data import clean_dataset, load_data
from coral_models.model_setup import load_model_setup
from coral_models.protocols import Processor


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Evaluate ASR models on a dataset"""
    dataset: DatasetDict | IterableDatasetDict = load_data(cfg)
    model_data = load_model_setup(cfg).load_saved()

    # Clean and tokenize the transcriptions
    dataset = clean_dataset(cfg, dataset=dataset)
    dataset = dataset.cast_column(
        column="audio", feature=Audio(sampling_rate=cfg.dataset.sampling_rate)
    )
    dataset = preprocess_transcriptions(
        dataset=dataset,
        processor=model_data.processor,
        text_column=cfg.dataset.text_column,
    )

    trainer = Trainer(
        args=TrainingArguments(".", remove_unused_columns=False, report_to=[]),
        model=model_data.model,
        data_collator=model_data.data_collator,
        compute_metrics=model_data.compute_metrics,
        eval_dataset=dataset,
        tokenizer=getattr(model_data.processor, "tokenizer"),
    )

    metrics = trainer.evaluate(dataset)
    wer = metrics["eval_wer"]

    print(f"\n*** RESULTS ON {cfg.dataset.name} ***")
    print(f"{cfg.hub_id} achieved a WER of {wer:.2%}.\n")


def preprocess_transcriptions(
    dataset: DatasetDict | IterableDatasetDict,
    processor: Processor,
    text_column: str = "sentence",
) -> DatasetDict | IterableDatasetDict:
    def tokenize_examples(example: dict) -> dict:
        example["labels"] = processor(
            text=example[text_column], truncation=True
        ).input_ids
        example["input_length"] = len(example["labels"])
        return example

    mapped = dataset.map(tokenize_examples)

    # After calling `map` the DatasetInfo is lost, so we need to add it back in
    for split in dataset.keys():
        mapped[split]._info.features["labels"] = Sequence(
            feature=Value(dtype="int64"), length=-1
        )
        mapped[split]._info.features["input_length"] = Value(dtype="int64")
        mapped[split]._info = dataset[split]._info

    return mapped


if __name__ == "__main__":
    main()
