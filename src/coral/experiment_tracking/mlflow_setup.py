"""MLFlow experiment tracking setup class."""

import os

import mlflow
from omegaconf import DictConfig

from .extracking_setup import ExTrackingSetup


class MLFlowSetup(ExTrackingSetup):
    """MLFlow setup class."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the MLFlow setup.

        Args:
            config:
                The configuration object.
        """
        self.config = config
        self.is_main_process = os.getenv("RANK", "0") == "0"

    def run_initialization(self) -> None:
        """Run the initialization of the experiment tracking setup."""
        mlflow.set_experiment(self.config.experiment_tracking.name_experiment)
        mlflow.start_run(run_name=self.config.experiment_tracking.name_run)
        return

    def run_finalization(self) -> None:
        """Run the finalization of the experiment tracking setup."""
        mlflow.end_run()
        return
