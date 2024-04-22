"""Reads the saved trainer state of a training checkpoint and plots training loss.

Usage:
    python src/scripts/plot_training_trajectory.py\
            <path/to/the/last/checkpoint> <path/to/plot/dir>
"""

import logging
from pathlib import Path

import click
from transformers import TrainerState
from transformers.trainer import TRAINER_STATE_NAME

from coral.plot import plot_training_loss

EXTENSION = "png"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.argument("checkpoint-dir", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path(exists=True))
@click.option(
    "--log-scale",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to produce the plot with log scaled X axis.",
)
def main(checkpoint_dir: str | Path, output_dir: str | Path, log_scale: bool) -> None:
    """Reads the saved trainer state of a training checkpoint and plots training curves.
    Currently, only a loss curve, loss.png, is produced to the output dir.
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)

    state = TrainerState.load_from_json(checkpoint_dir / TRAINER_STATE_NAME)

    loss_plot_path = output_dir / f"loss.{EXTENSION}"
    plot_training_loss(state, loss_plot_path, log_scale=log_scale)


if __name__ == "__main__":
    main()
