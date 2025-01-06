"""Functions related to the model setups."""

from omegaconf import DictConfig

from .data_models import ModelSetup
from .wav2vec2 import Wav2Vec2ModelSetup
from .whisper import WhisperModelSetup


def load_model_setup(config: DictConfig) -> ModelSetup:
    """Get the model setup for the given configuration.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        The model setup.
    """
    model_type: str = config.model.type
    match model_type:
        case "wav2vec2":
            return Wav2Vec2ModelSetup(config=config)
        case "whisper":
            return WhisperModelSetup(config=config)
        case _:
            raise ValueError(f"Unsupported model type: {model_type!r}")
