"""Functions related to the preprocessing of the data for Wav2Vec 2.0 models."""

from datasets import Audio, DatasetDict
from omegaconf import DictConfig
from transformers import (
    PreTrainedTokenizerBase,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)


class ModifiedWav2Vec2Processor(Wav2Vec2Processor):
    """The preprocessor for the Wav2Vec 2.0 model.

    This class extends the `Wav2Vec2Processor` class from the Hugging Face
    `transformers` package, adding the `preprocess` method to preprocess the audio and
    transcriptions in a dataset.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        self.cfg = cfg
        self.tokenizer: PreTrainedTokenizerBase = kwargs["tokenizer"]
        super().__init__(**kwargs)

    def preprocess(self, dataset: DatasetDict) -> DatasetDict:
        """Preprocess the audio and transcriptions in a dataset.

        Args:
            dataset (DatasetDict):
                The dataset containing the audio and transcriptions.

        Returns:
            DatasetDict:
                The preprocessed audio and transcriptions dataset.
        """
        dataset = self.preprocess_audio(dataset=dataset)
        dataset = self.preprocess_transcriptions(dataset=dataset)
        return dataset

    def preprocess_audio(self, dataset: DatasetDict) -> DatasetDict:
        """Preprocess the audio feature in a dataset.

        Args:
            dataset (DatasetDict):
                The dataset containing the audio feature.

        Returns:
            DatasetDict:
                The preprocessed audio dataset.
        """
        audio = Audio(sampling_rate=self.cfg.model.sampling_rate)
        return dataset.cast_column("audio", audio)

    def preprocess_transcriptions(self, dataset: DatasetDict) -> DatasetDict:
        """Preprocess the transcriptions in a dataset.

        Args:
            dataset (DatasetDict):
                The dataset containing the transcriptions.

        Returns:
            DatasetDict:
                The preprocessed transcriptions dataset.
        """

        def tokenize_examples(example: dict) -> dict:
            example["labels"] = self(
                text=example[self.cfg.dataset.text_column]
            ).input_ids
            example["input_length"] = len(example["labels"])
            return example

        return dataset.map(tokenize_examples)


def load_processor(cfg: DictConfig) -> ModifiedWav2Vec2Processor:
    """Load the processor for a Wav2Vec 2.0 model.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.

    Returns:
        ModifiedWav2Vec2Processor:
            The processor for a Wav2Vec 2.0 model.
    """
    # Initialise the tokenizer
    tokenizer: Wav2Vec2CTCTokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        cfg.model_dir,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        word_delimiter_token="|",
    )

    # Set the `model_max_length` attribute of the tokenizer, if it hasn't been set
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 1e6:
        tokenizer.model_max_length = 512

    # Initialise the feature extractor
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=cfg.model.sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    # Initialise the processor, which wraps the tokenizer and the extractor
    processor = ModifiedWav2Vec2Processor(
        cfg, feature_extractor=extractor, tokenizer=tokenizer
    )

    return processor
