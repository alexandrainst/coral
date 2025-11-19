"""Utility functions for Hugging Face Trainers."""

from collections.abc import Callable

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.trainer import Trainer
from transformers.trainer_seq2seq import Seq2SeqTrainer


class TrainerWithMultipleDataCollators(Trainer):
    """A Trainer that uses different data collators for training and evaluation."""

    def __init__(self, get_data_collator_fn: Callable, **kwargs) -> None:
        """Initialise the Trainer.

        Args:
            get_data_collator_fn:
                The function which returns the data collator. Needs to have a `training`
                Boolean argument.
            **kwargs:
                All other regular Trainer initialisation arguments.
        """
        self.get_data_collator_fn = get_data_collator_fn
        super().__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns:
            The training dataloader.
        """
        self.data_collator = self.get_data_collator_fn(training=True)
        return super().get_train_dataloader()

    def get_eval_dataloader(
        self, eval_dataset: str | Dataset | None = None
    ) -> DataLoader:
        """Get the evaluation dataloader.

        Args:
            eval_dataset (optional):
                The evaluation dataset.

        Returns:
            The dataloader.
        """
        self.data_collator = self.get_data_collator_fn(training=False)
        return super().get_eval_dataloader(eval_dataset=eval_dataset)


class Seq2SeqTrainerWithMultipleDataCollators(Seq2SeqTrainer):
    """A Seq2SeqTrainer that uses different collators for training and evaluation."""

    def __init__(self, get_data_collator_fn: Callable, **kwargs) -> None:
        """Initialise the Trainer.

        Args:
            get_data_collator_fn:
                The function which returns the data collator. Needs to have a `training`
                Boolean argument.
            **kwargs:
                All other regular Trainer initialisation arguments.
        """
        self.get_data_collator_fn = get_data_collator_fn
        super().__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns:
            The training dataloader.
        """
        self.data_collator = self.get_data_collator_fn(training=True)
        return super().get_train_dataloader()

    def get_eval_dataloader(
        self, eval_dataset: str | Dataset | None = None
    ) -> DataLoader:
        """Get the evaluation dataloader.

        Args:
            eval_dataset (optional):
                The evaluation dataset.

        Returns:
            The dataloader.
        """
        self.data_collator = self.get_data_collator_fn(training=False)
        return super().get_eval_dataloader(eval_dataset=eval_dataset)
