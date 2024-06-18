"""Evaluate a speech model.

Usage:
    python evaluate_model.py <key>=<value> <key>=<value> ...
"""

import logging

import hydra
from coral.evaluate import evaluate
from dotenv import load_dotenv
from omegaconf import DictConfig

load_dotenv()


logger = logging.getLogger("coral")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Evaluate a speech model on a dataset.

    Args:
        cfg:
            The Hydra configuration object.
    """
    score_df = evaluate(cfg=cfg)
    score_df.to_csv(f"{cfg.pipeline_id}_scores.csv", index=False)


if __name__ == "__main__":
    main()
