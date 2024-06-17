"""Evaluate a speech model.

Usage:
    python evaluate_model.py <key>=<value> <key>=<value> ...
"""

import hydra
from coral.data import load_data
from coral.model_setup import load_model_setup
from coral.protocols import Processor
from coral.utils import transformers_output_ignored
from datasets import DatasetDict, IterableDatasetDict, Sequence, Value
from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

load_dotenv()


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Evaluate a speech model on a dataset.

    Args:
        cfg:
            The Hydra configuration object.
    """
    eval_dataset_cfg = list(cfg.datasets.values())[0]

    with transformers_output_ignored():
        model_data = load_model_setup(cfg).load_saved()

    dataset: DatasetDict | IterableDatasetDict = load_data(cfg)
    dataset = preprocess_transcriptions(dataset=dataset, processor=model_data.processor)

    trainer = Trainer(
        args=TrainingArguments(".", remove_unused_columns=False, report_to=[]),
        model=model_data.model,
        data_collator=model_data.data_collator,
        compute_metrics=model_data.compute_metrics,
        eval_dataset=dataset,
        tokenizer=getattr(model_data.processor, "tokenizer"),
    )

    metrics = trainer.evaluate(eval_dataset=dataset["test"])
    wer = metrics["eval_wer"]

    print(f"\n*** RESULTS ON {eval_dataset_cfg.dataset.name} ***")
    print(f"{eval_dataset_cfg.hub_id} achieved a WER of {wer:.2%}.\n")


def preprocess_transcriptions(
    dataset: DatasetDict | IterableDatasetDict, processor: Processor
) -> DatasetDict | IterableDatasetDict:
    """Preprocess the transcriptions in the dataset.

    Args:
        dataset:
            The dataset to preprocess.
        processor:
            The processor to use for tokenization.

    Returns:
        The preprocessed dataset.
    """

    def tokenize_examples(example: dict) -> dict:
        example["labels"] = processor(text=example["text"], truncation=True).input_ids
        example["input_length"] = len(example["labels"])
        return example

    mapped = dataset.map(tokenize_examples)

    # After calling `map` the DatasetInfo is lost, so we need to add it back in
    for split in dataset.keys():
        mapped[split]._info = dataset[split]._info
        mapped[split]._info.features["labels"] = Sequence(
            feature=Value(dtype="int64"), length=-1
        )
        mapped[split]._info.features["input_length"] = Value(dtype="int64")
        mapped[split]._info = dataset[split]._info

    return mapped


if __name__ == "__main__":
    main()
