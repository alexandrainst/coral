"""Evaluate a speech model.

Usage:
    python src/scripts/evaluate_model.py <key>=<value> <key>=<value> ...
"""

import logging

import hydra
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
    score_df.to_csv(f"{config.model_id}_scores.csv", index=False)


if __name__ == "__main__":
    main()
