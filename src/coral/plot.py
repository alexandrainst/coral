"""Functions related to visualisation."""

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from transformers import TrainerState


def plot_training_loss(
    trainer_state: TrainerState,
    output_path: Path,
    title: str = "ASR Training loss",
    log_scale: bool = False,
) -> None:
    """Plot the training loss over steps from a trainer's state.

    Args:
        trainer_state:
            The state object that holds logs and other training related information.
        output_path:
            The path where the generated plot should be saved.
        title:
            The title for the plot. Defaults to 'ASR TRAINING loss'.
        log_scale:
            If set to True, the y-axis will be set to logarithmic scale. Defaults to
            False.
    """
    x = [log["step"] for log in trainer_state.log_history]
    y = [log["loss"] for log in trainer_state.log_history]

    fig, ax = plt.subplots()
    assert isinstance(ax, Axes)
    assert isinstance(fig, Figure)

    if log_scale:
        ax.set_yscale("log")

    ax.plot(x, y)

    ax.set_xlabel(xlabel="Steps")
    ax.set_ylabel(ylabel="Loss")
    ax.set_title(label=title)
    ax.grid(visible=True)

    fig.savefig(fname=output_path)
