"""This module contains the base class for an experiment tracking setup."""

from abc import ABC, abstractmethod

from omegaconf import DictConfig


class ExTrackingSetup(ABC):
    """Base class for an experiment tracking setup."""

    @abstractmethod
    def __init__(self, config: DictConfig) -> None:
        """Initialise the experiment tracking setup.

        Args:
            config:
                The configuration object.
        """

    @abstractmethod
    def run_initialization(self) -> bool:
        """Run the initialization of the experiment tracking setup.

        Returns:
            True if the initialization was successful, False otherwise.
        """

    @abstractmethod
    def run_finalization(self) -> bool:
        """Run the finalization of the experiment tracking setup.

        Returns:
            True if the finalization was successful, False otherwise.
        """
