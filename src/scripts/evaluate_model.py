"""Evaluate a speech model.

Usage:
    python src/scripts/evaluate_model.py <key>=<value> <key>=<value> ...
"""

import logging
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig

from coral.evaluate import evaluate

load_dotenv()


logger = logging.getLogger("coral")


@hydra.main(config_path="../../config", config_name="evaluation", version_base=None)
def main(config: DictConfig) -> None:
    """Evaluate a speech model on a dataset.

    Args:
        config:
            The Hydra configuration object.
    """
    score_df = evaluate(config=config)
    if config.store_results:
        model_id_without_slashes = config.model_id.replace("/", "__")
        if "coral" in config.dataset:
            filename = Path(f"{model_id_without_slashes}_coral_scores.csv")
            score_df.to_csv(filename, index=False)
        else:
            filename = Path(f"{model_id_without_slashes}_scores.csv")
            if filename.exists():
                existing_score_df = pd.read_csv(filename)
                score_df = pd.merge(
                    left=existing_score_df,
                    right=score_df,
                    how="outer",
                    left_index=True,
                    right_index=True,
                )
            score_df.to_csv(filename)


if __name__ == "__main__":
    main()
