"""Factory for experiment tracking setup."""

from omegaconf import DictConfig

from .extracking_setup import ExTrackingSetup
from .mlflow_setup import MLFlowSetup
from .wandb_setup import WandbSetup


def load_extracking_setup(config: DictConfig) -> ExTrackingSetup:
    """Return the experiment tracking setup.

    Args:
        config:
            The configuration object.

    Returns:
        The experiment tracking setup.
    """
    match config.experiment_tracking.type:
        case "wandb":
            return WandbSetup(config=config)
        case "mlflow":
            return MLFlowSetup(config=config)
        case _:
            raise ValueError(
                f"Unknown experiment tracking type: {config.experiment_tracking.type}"
            )
