"""Finetune a Qwen3-TTS speech synthesis model on the CoRal-TTS dataset."""

import importlib.util
import typing as t

from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset

if importlib.util.find_spec("qwen_tts") is not None:
    from qwen_tts import Qwen3TTSTokenizer


def prepare_data(dataset_id: str, speaker: t.Literal["mic", "nic"]) -> Dataset:
    """Prepare the data for training.

    Args:
        dataset_id:
            The dataset ID.
        speaker:
            The speaker to use.

    Returns:
        The prepared dataset.
    """
    # Load the dataset
    dataset = load_dataset(path=dataset_id, split="train")
    assert isinstance(dataset, Dataset), (f"Expected a Dataset, got {type(dataset)}",)

    # Filter the dataset
    dataset = dataset.filter(lambda x: x["speaker_id"] == speaker)
    dataset = dataset.filter(lambda x: x["audio"]["array"].shape[0] > 0)

    # Tokenise the audio
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="auto"
    )
    dataset = dataset.map(
        lambda batch: dict(
            audio_codes=tokenizer.encode(
                audios=[x["array"] for x in batch["audio"]],
                sr=batch["audio"][0]["sampling_rate"],
            ).audio_codes
        ),
        batched=True,
        batch_size=32,
    )
    assert isinstance(dataset, Dataset), (
        f"Expected a Dataset after tokenisation, got {type(dataset)}",
    )

    return dataset
