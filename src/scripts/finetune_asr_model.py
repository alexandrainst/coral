"""Finetune a speech recognition model.

Usage:
    Running on a single GPU:

    python src/scripts/finetune_asr_model.py \
        [key=value] [key=value] ...

    Running on multiple GPUs:

    accelerate launch [--use-deepspeed] src/scripts/finetune_asr_model.py \
        [key=value] [key=value] ...
"""

import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from coral.finetune import finetune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("coral_finetuning")


load_dotenv()


@hydra.main(config_path="../../config", config_name="asr_finetuning", version_base=None)
def main(config: DictConfig) -> None:
    """Finetune an ASR model.

    Args:
        config:
            The Hydra configuration object.
    """
    finetune(config=config)


if __name__ == "__main__":
    main()
