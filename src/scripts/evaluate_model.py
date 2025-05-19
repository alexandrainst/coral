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

    for dataset_name, dataset_config  in config['datasets'].items():

        df_scores, df_predictions = evaluate(config=config, dataset_config=dataset_config)

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
            # prepare the output names and directory
            dataset_without_slashes = dataset_config.id.replace("/", "-")
            results_dir = Path("outputs") / Path("results") / dataset_without_slashes
            results_dir.mkdir(exist_ok=True, parents=True)

            model_id_without_slashes = config.model_id.replace("/", "--")

            # Save evaluation scores
            if config.detailed:
                path_results = results_dir / Path(f"{model_id_without_slashes}-{dataset_name}-scores.csv")
                df_scores.to_csv(path_results, index=False)
            else:
                path_results = results_dir / Path("evaluation-results.csv")
                if path_results.exists():
                    df_scores = pd.concat(
                        objs=[pd.read_csv(path_results, index_col=False), df_scores],
                        ignore_index=True,
                    )
                df_scores.to_csv(path_results, index=False)

            # Save predictions
            path_predictions = results_dir / Path(
                f"{model_id_without_slashes}-{dataset_name}-predictions.csv"
            )
            df_predictions.to_csv(path_predictions, index=False)


if __name__ == "__main__":
    main()
