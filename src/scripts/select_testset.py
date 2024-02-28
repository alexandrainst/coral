"""Script for selecting a test set from the full dataset.

Usage:
    python select_testset.py +datasets=coral_test_set
"""

import hydra
import pandas as pd
import logging
from pathlib import Path
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Selects the test set from the full CoRal dataset."""
    data_dir = Path(cfg.dirs.data)
    processed_data_path = data_dir / cfg.dirs.processed
    speaker_metadata_path = data_dir / cfg.dirs.hidden / "speakers.xlsx"
    processed_audio_path = processed_data_path / "processed_audio"
    recordings_metadata_path = processed_data_path / "recordings.xlsx"

    if not processed_audio_path.exists():
        raise FileNotFoundError(f"{processed_audio_path} does not exist")
    if not speaker_metadata_path.exists():
        raise FileNotFoundError(f"{speaker_metadata_path} does not exist")
    if not recordings_metadata_path.exists():
        raise FileNotFoundError(f"{recordings_metadata_path} does not exist")

    # Load the metadata
    pd.read_excel(speaker_metadata_path)
    pd.read_excel(recordings_metadata_path)


if __name__ == "__main__":
    main()
