"""Functions related to visualization"""

from pathlib import Path

from matplotlib import pyplot as plt
from transformers import TrainerState

DEFAULT_LOSS_TITLE = "ASR Training loss"


def plot_training_loss(
    trainer_state: TrainerState,
    output_path: Path,
    title: str = DEFAULT_LOSS_TITLE,
    log_scale: bool = False,
) -> None:
    """
    Plot the training loss over steps from a trainer's state.

    Args:
        trainer_state (TrainerState):
            The state object that holds logs and other training related information.
        output_path (Path):
            The path where the generated plot should be saved.
        title (str, optional):
            The title for the plot. Defaults to 'ASR TRAINING loss'.
        log_scale (bool, optional):
            If set to True, the y-axis will be set to logarithmic scale.
                Defaults to False.
    """
    x = [log["step"] for log in trainer_state.log_history]
    y = [log["loss"] for log in trainer_state.log_history]

    fig, ax = plt.subplots()
    if log_scale:
        ax.set_yscale("log")

    ax.plot(x, y)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True)

    fig.savefig(output_path)
