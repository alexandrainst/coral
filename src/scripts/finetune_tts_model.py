"""Finetune a text synthesis model.

Usage:
    Running on a single GPU:

    python src/scripts/finetune_tts_model.py

    Running on multiple GPUs:

    accelerate launch [--use-deepspeed] src/scripts/finetune_asr_model.py
"""

import logging
from argparse import ArgumentParser

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
    """Finetune an TTS model."""
    parser = ArgumentParser()
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional number of samples to load for quick debugging.",
    )
    args = parser.parse_args()

    dataset = prepare_data(
        dataset_id="CoRal-project/coral-tts",
        speaker="mic",
        max_samples=args.max_samples,
    )
    finetune_tts_model(dataset=dataset)


if __name__ == "__main__":
    main()
