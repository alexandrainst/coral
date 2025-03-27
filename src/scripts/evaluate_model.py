"""Evaluate a speech model.

Usage:
    python src/scripts/evaluate_model.py [key=value] [key=value] ...
"""

import logging
from pathlib import Path
from shutil import rmtree

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig

from coral.evaluate import evaluate

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("coral_evaluation")


@hydra.main(config_path="../../config", config_name="evaluation", version_base=None)
def main(config: DictConfig) -> None:
    """Evaluate a speech model on a dataset.

    Args:
        config:
            The Hydra configuration object.
    """
    score_df = evaluate(config=config)

    # Loading the pipeline stores it to the default HF cache, and they don't allow
    # changing it for pipelines. So we remove the models stored in the cache manually,
    # to avoid running out of disk space.
    model_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dirs = list(
        model_cache_dir.glob(f"model--{config.model_id.replace('/', '--')}")
    )
    if model_dirs:
        for model_dir in model_dirs:
            rmtree(path=model_dir)

    if config.store_results:
        dataset_without_slashes = config.dataset.replace("/", "-").replace("::", "-")
        results_dir = Path("results") / dataset_without_slashes
        results_dir.mkdir(parents=True, exist_ok=True)

        model_id_without_slashes = config.model_id.replace("/", "--")
        if "coral" in config.dataset and config.detailed:
            path_results = results_dir / Path(f"{model_id_without_slashes}-scores.csv")
            score_df.to_csv(path_results, index=False)
        else:
            path_results = results_dir / Path("evaluation-results.csv")
            if path_results.exists():
                score_df = pd.concat(
                    objs=[pd.read_csv(path_results, index_col=False), score_df],
                    ignore_index=True,
                )
            score_df.to_csv(path_results, index=False)


if __name__ == "__main__":
    main()
