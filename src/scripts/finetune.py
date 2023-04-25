"""Script that finetunes a speech model.

Usage:
    python finetune.py <key>=<value> <key>=<value> ...
"""

import warnings

import hydra
from omegaconf import DictConfig

from coral_models import finetune_wav2vec2, finetune_whisper


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Finetune an ASR model.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
    """
    # Ignore user warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Finetune
    if cfg.model.type == "wav2vec2":
        finetune_wav2vec2(cfg)
    elif cfg.model.type == "whisper":
        finetune_whisper(cfg)
    else:
        raise NotImplementedError(f"Unsupported model type: {cfg.model.name}")


if __name__ == "__main__":
    main()
