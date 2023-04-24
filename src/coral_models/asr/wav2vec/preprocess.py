"""Functions related to the preprocessing of the data for Wav2Vec 2.0 models."""

from datasets import DatasetDict
from omegaconf import DictConfig
from transformers import Wav2Vec2Processor


class Processor:
    """The preprocessor for the Wav2Vec 2.0 model.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.wav2vec2_processor = self.load_wav2vec2_processor()

    def load_wav2vec2_processor(self) -> Wav2Vec2Processor:
        """Load the preprocessor for the Wav2Vec 2.0 model.

        Returns:
            Wav2Vec2Processor:
                The preprocessor for the Wav2Vec 2.0 model.
        """
        raise NotImplementedError

    def preprocess_audio(self, dataset: DatasetDict) -> DatasetDict:
        """Preprocess the audio feature in a dataset.

        Args:
            dataset (DatasetDict):
                The dataset containing the audio feature.

        Returns:
            DatasetDict:
                The preprocessed audio dataset.
        """
        raise NotImplementedError

    def preprocess_transcriptions(self, dataset: DatasetDict) -> DatasetDict:
        """Preprocess the transcriptions in a dataset.

        Args:
            dataset (DatasetDict):
                The dataset containing the transcriptions.

        Returns:
            DatasetDict:
                The preprocessed transcriptions dataset.
        """
        raise NotImplementedError
