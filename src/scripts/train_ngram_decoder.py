"""Train an n-gram language model for the decoder of a finetuned Wav2Vec 2.0 model.

Usage:
    python src/scripts/train_ngram_decoder.py <key>=<value> <key>=<value> ...
"""

import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import hydra
import requests
from datasets import Dataset, load_dataset
from huggingface_hub import Repository
from omegaconf import DictConfig
from pyctcdecode.decoder import build_ctcdecoder
from transformers import AutoProcessor, Wav2Vec2ProcessorWithLM


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def train_ngram_model(config: DictConfig) -> None:
    """Trains an ngram language model.

    Args:
        config:
            Hydra configuration dictionary.
    """
    dataset = load_dataset(
        path=config.model.decoder.dataset_id,
        name=config.model.decoder.dataset_subset,
        split=config.model.decoder.dataset_split,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
    )
    assert isinstance(dataset, Dataset)

    # Ensure that the `kenlm` directory exists, and download if otherwise
    kenlm_dir = Path.home() / ".cache" / "kenlm"
    if not kenlm_dir.exists():
        download_and_extract(
            url="https://kheafield.com/code/kenlm.tar.gz", target_dir=kenlm_dir
        )

    # Compile `kenlm` if it hasn't already been compiled
    kenlm_build_dir = kenlm_dir / "build"
    if not kenlm_build_dir.exists():
        kenlm_build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmake", ".."], cwd=str(kenlm_dir / "build"))
        subprocess.run(["make", "-j", "2"], cwd=str(kenlm_dir / "build"))

    # Train the n-gram language model if it doesn't already exist
    correct_ngram_path = Path(config.model_dir) / f"{config.model.decoder.n}gram.arpa"
    if not correct_ngram_path.exists():
        ngram_path = Path(config.model_dir) / f"raw_{config.model.decoder.n}gram.arpa"
        ngram_path.parent.mkdir(parents=True, exist_ok=True)

        # If the raw language model does not exist either then train from scratch
        if not ngram_path.exists():
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as text_file:
                # Dump dataset to a temporary text file
                text_file.write("\n\n".join(dataset["text"]))
                text_file.flush()

                # Train the n-gram language model
                with ngram_path.open("w") as f_out:
                    subprocess.run(
                        ["kenlm/build/bin/lmplz", "-o", str(config.model.decoder.n)],
                        stdin=text_file,
                        stdout=f_out,
                    )

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

    processor = AutoProcessor.from_pretrained(config.model_dir)

    # Extract the vocabulary, which will be used to build the CTC decoder
    vocab_dict: dict[str, int] = processor.tokenizer.get_vocab()
    sorted_vocab_list = sorted(vocab_dict.items(), key=lambda item: item[1])
    sorted_vocab_dict = {k.lower(): v for k, v in sorted_vocab_list}

    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()), kenlm_model_path=str(correct_ngram_path)
    )

    # Build the processor with LM included
    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder,
    )

    # Clone the repo containing the finetuned model
    repo = Repository(local_dir=config.model_dir, clone_from=config.model_dir)

    # Save the new processor to the repo
    processor_with_lm.save_pretrained(config.model_dir)

    # Compress the ngram model
    subprocess.run(
        [
            "kenlm/build/bin/build_binary",
            str(correct_ngram_path),
            str(correct_ngram_path.with_suffix(".bin")),
        ]
    )

    # Remove the uncompressed ngram model
    if correct_ngram_path.exists():
        correct_ngram_path.unlink()

    # Push the changes to the repo
    repo.push_to_hub(commit_message="Upload LM-boosted decoder")


def download_and_extract(url: str, target_dir: str | Path) -> None:
    """Download and extract a compressed file from a URL.

    Args:
        url (str):
            URL to download from.
        target_dir (str | Path):
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


if __name__ == "__main__":
    train_ngram_model()
