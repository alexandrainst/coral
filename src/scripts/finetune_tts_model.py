"""Finetune a text synthesis model.

Usage:
    Running on a single GPU:

    python src/scripts/finetune_tts_model.py

    Running on multiple GPUs:

    accelerate launch [--use-deepspeed] src/scripts/finetune_asr_model.py
"""

import logging

from dotenv import load_dotenv

from coral.tts import finetune_tts_model, prepare_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("coral_finetuning")


load_dotenv()


def main() -> None:
    """Finetune an ASR model.

    Args:
        config:
            The Hydra configuration object.
    """
    dataset = prepare_data(dataset_id="coral/coral-tts", speaker="mic")
    finetune_tts_model(dataset=dataset)


if __name__ == "__main__":
    main()
