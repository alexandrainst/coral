"""Training n-gram language model for Wav2Vec2 models."""

import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import requests
from datasets import Dataset, enable_progress_bar, load_dataset
from omegaconf import DictConfig
from pyctcdecode.decoder import build_ctcdecoder
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

from .data import process_dataset


def train_ngram_model(config: DictConfig) -> None:
    """Trains an ngram language model.

    Args:
        config:
            Hydra configuration dictionary.
    """
    # Ensure that the `kenlm` directory exists, and download if otherwise
    cache_dir = Path.home() / ".cache"
    kenlm_dir = cache_dir / "kenlm"
    if not kenlm_dir.exists():
        download_and_extract(
            url="https://kheafield.com/code/kenlm.tar.gz", target_dir=cache_dir
        )

    # Compile `kenlm` if it hasn't already been compiled
    kenlm_build_dir = kenlm_dir / "build"
    if not (kenlm_build_dir / "bin" / "lmplz").exists():
        kenlm_build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmake", ".."], cwd=str(kenlm_build_dir))
        subprocess.run(["make", "-j", "2"], cwd=str(kenlm_build_dir))

    # Train the n-gram language model if it doesn't already exist
    correct_ngram_path = Path(config.model_dir) / f"{config.model.decoder.n}gram.arpa"
    if not correct_ngram_path.exists():
        ngram_path = Path(config.model_dir) / f"raw_{config.model.decoder.n}gram.arpa"
        ngram_path.parent.mkdir(parents=True, exist_ok=True)

        # If the raw language model does not exist either then train from scratch
        if not ngram_path.exists():
            dataset = load_dataset(
                path=config.model.decoder.dataset_id,
                name=config.model.decoder.dataset_subset,
                split=config.model.decoder.dataset_split,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
            )
            assert isinstance(dataset, Dataset)

            enable_progress_bar()
            dataset = process_dataset(
                dataset=dataset,
                characters_to_keep=config.characters_to_keep,
                text_column=config.model.decoder.text_column,
                audio_column=None,
                lower_case=True,
            )
            assert isinstance(dataset, Dataset)

            # Deduplicating the sentences in the dataset is required when training the
            # n-gram language model
            sentences = list(set(dataset[config.model.decoder.text_column]))

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as text_file:
                # Dump dataset to a temporary text file
                text_file.write("\n".join(sentences))
                text_file.flush()

                # Train the n-gram language model
                with Path(text_file.name).open() as f_in, ngram_path.open("w") as f_out:
                    subprocess.run(
                        [
                            str(kenlm_build_dir / "bin" / "lmplz"),
                            "-o",
                            str(config.model.decoder.n),
                        ],
                        stdin=f_in,
                        stdout=f_out,
                    )

                assert ngram_path.exists(), "Failed to train n-gram language model"

        # Add end-of-sentence marker </s> to the n-gram language model to get the final
        # language model
        with ngram_path.open("r") as f_in:
            with correct_ngram_path.open("w") as f_out:
                has_added_eos = False
                for line in f_in:
                    # Increment the 1-gram count by 1
                    if not has_added_eos and "ngram 1=" in line:
                        count = line.strip().split("=")[-1]
                        new_line = line.replace(f"{count}", f"{int(count)+1}")
                        f_out.write(new_line)

                    # Add the end-of-sentence marker right after the the
                    # start-of-sentence marker
                    elif not has_added_eos and "<s>" in line:
                        f_out.write(line)
                        f_out.write(line.replace("<s>", "</s>"))
                        has_added_eos = True

                    # Otherwise we're just copying the line verbatim
                    else:
                        f_out.write(line)

        # Remove non-correct ngram model again
        if ngram_path.exists():
            ngram_path.unlink()

    processor = Wav2Vec2Processor.from_pretrained(config.model_dir)

    # Extract the vocabulary, which will be used to build the CTC decoder
    vocab_dict: dict[str, int] = processor.tokenizer.get_vocab()
    sorted_vocab_list = sorted(vocab_dict.items(), key=lambda item: item[1])
    sorted_vocab_dict = {k.lower(): v for k, v in sorted_vocab_list}

    # Build the processor with LM included and save it
    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()), kenlm_model_path=str(correct_ngram_path)
    )
    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder,
    )
    processor_with_lm.save_pretrained(config.model_dir)

    # Remove the ngram model again, as the `save_pretrained` method also saves the
    # ngram model
    if correct_ngram_path.exists():
        correct_ngram_path.unlink()

    # Compress the ngram model
    new_ngram_path = (
        Path(config.model_dir) / "language_model" / f"{config.model.decoder.n}gram.arpa"
    )
    subprocess.run(
        [
            str(kenlm_build_dir / "bin" / "build_binary"),
            str(new_ngram_path),
            str(new_ngram_path.with_suffix(".bin")),
        ]
    )

    if config.push_to_hub:
        processor_with_lm.push_to_hub(
            repo_id=f"{config.hub_organisation}/{config.model_id}"
        )

    # Remove the uncompressed ngram model, as we only need the compressed version
    if new_ngram_path.exists():
        new_ngram_path.unlink()


def download_and_extract(url: str, target_dir: str | Path) -> None:
    """Download and extract a compressed file from a URL.

    Args:
        url:
            URL to download from.
        target_dir:
            Path to the directory where the file should be downloaded to.
    """
    # Download the file and load the data as bytes into memory
    with requests.get(url) as response:
        status_code: int = response.status_code  # type: ignore[attr-defined]
        if status_code != 200:
            raise requests.HTTPError(f"Received status code {status_code} from {url}")
        data = response.content  # type: ignore[attr-defined]

    # Extract the file
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(path=target_dir)
