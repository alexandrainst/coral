"""wandb experiment tracking setup class."""

import os
from wandb import finish as wandb_finish
from wandb.sdk.wandb_init import init as wandb_init

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

    def run_initialization(self) -> bool:
        """Run the initialization of the experiment tracking setup."""
        wandb_init(
            project=self.config.experiment_tracking.name_experiment,
            name=self.config.experiment_tracking.name_run,
            group=self.config.experiment_tracking.name_group,
            config=dict(self.config),
        )
        return True
    
    def run_finalization(self) -> bool:
        """Run the finalization of the experiment tracking setup."""
        wandb_finish()
        return True