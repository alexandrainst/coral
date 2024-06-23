"""Train an n-gram language model for the decoder of a finetuned Wav2Vec 2.0 model.

Usage:
    python src/scripts/train_ngram_decoder.py <key>=<value> <key>=<value> ...
"""

import hydra
from coral.ngram import train_ngram_model
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Trains an ngram language model.

    Args:
        config:
            Hydra configuration dictionary.
    """
    train_ngram_model(config=config)


if __name__ == "__main__":
    main()
