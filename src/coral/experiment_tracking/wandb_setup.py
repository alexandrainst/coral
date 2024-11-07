"""wandb experiment tracking setup class."""

import os

import wandb
from omegaconf import DictConfig

from .extracking_setup import ExTrackingSetup


class WandbSetup(ExTrackingSetup):
    """Wandb setup class."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the Wandb setup.

        Args:
            config:
                The configuration object.
        """
        self.config = config
        self.is_main_process = os.getenv("RANK", "0") == "0"

    def run_initialization(self) -> None:
        """Run the initialization of the experiment tracking setup."""
        wandb.init(
            project=self.config.experiment_tracking.name_experiment,
            name=self.config.experiment_tracking.name_run,
            group=self.config.experiment_tracking.name_group,
            config=dict(self.config),
        )
        return

    def run_finalization(self) -> None:
        """Run the finalization of the experiment tracking setup."""
        wandb.finish()
        return
