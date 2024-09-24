"""Training n-gram language model for Wav2Vec2 models."""

import io
import logging
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import requests
from datasets import Dataset, IterableDataset, interleave_datasets, load_dataset
from omegaconf import DictConfig
from pyctcdecode.decoder import build_ctcdecoder
from tqdm.auto import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

from coral.utils import convert_iterable_dataset_to_dataset

from .data import load_dataset_for_evaluation, process_dataset

logger = logging.getLogger(__package__)


def train_ngram_model(config: DictConfig) -> None:
    """Trains an ngram language model.

    Args:
        config:
            Hydra configuration dictionary.
    """
    is_main_process = os.getenv("RANK", "0") == "0"
    if not is_main_process:
        return

    # Ensure that the `kenlm` directory exists, and download if otherwise
    cache_dir = (
        Path.home() / ".cache" if config.cache_dir is None else Path(config.cache_dir)
    )
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
    correct_ngram_path = (
        Path(config.model_dir) / f"{config.model.decoder_num_ngrams}gram.arpa"
    )
    if not correct_ngram_path.exists():
        logger.info("Training n-gram language model...")

        ngram_path = (
            Path(config.model_dir) / f"raw_{config.model.decoder_num_ngrams}gram.arpa"
        )
        ngram_path.parent.mkdir(parents=True, exist_ok=True)

        # If the raw language model does not exist either then train from scratch
        if not ngram_path.exists():
            all_datasets: list[Dataset] = list()
            for dataset_name, dataset_config in config.decoder_datasets.items():
                logger.info(f"Loading dataset {dataset_name!r}")

                dataset = load_dataset(
                    path=dataset_config.id,
                    name=dataset_config.subset,
                    split=dataset_config.split,
                    token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
                    trust_remote_code=True,
                    cache_dir=config.cache_dir,
                    streaming=True,
                )
                assert isinstance(dataset, IterableDataset)

                if dataset_config.audio_column is not None:
                    dataset = dataset.remove_columns(
                        column_names=dataset_config.audio_column
                    )

                if dataset_config.text_column != "text":
                    dataset = dataset.rename_column(dataset_config.text_column, "text")

                dataset = process_dataset(
                    dataset=dataset,
                    clean_text=config.model.clean_text,
                    characters_to_keep=config.characters_to_keep,
                    text_column="text",
                    remove_input_dataset_columns=False,
                    audio_column=None,
                    convert_numerals=False,
                    lower_case=config.model.lower_case,
                )
                assert isinstance(dataset, IterableDataset)

                dataset = convert_iterable_dataset_to_dataset(
                    iterable_dataset=dataset, cache_dir=config.cache_dir
                )
                assert isinstance(dataset, Dataset)

                all_datasets.append(dataset)

            dataset = interleave_datasets(datasets=all_datasets)

            # Deduplicating the sentences in the dataset is required when training the
            # n-gram language model
            sentences = list(set(dataset[config.model.decoder.text_column]))

            # Remove sentences, that appear in the CoRal test split
            evaluation_config = DictConfig(
                dict(
                    dataset="alexandrainst/coral::read_aloud",
                    cache_dir=config.cache_dir,
                    eval_split_name="test",
                    text_column="text",
                    audio_column="audio",
                    sampling_rate=16_000,
                    min_seconds_per_example=0.5,
                    max_seconds_per_example=10,
                    clean_text=config.model.clean_text,
                    lower_case=config.model.lower_case,
                    characters_to_keep="abcdefghijklmnopqrstuvwxyzæøå0123456789éü",
                )
            )
            evaluation_dataset = load_dataset_for_evaluation(config=evaluation_config)
            evaluation_sentences = set(
                evaluation_dataset[evaluation_config.text_column]
            )
            sentences = [
                sentence.replace(evaluation_sentence, "")
                for sentence in tqdm(sentences, desc="Removing evaluation sentences")
                for evaluation_sentence in evaluation_sentences
                if evaluation_sentence in sentence
            ]

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
                            str(config.model.decoder_num_ngrams),
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

    logger.info("Storing n-gram language model...")

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
        Path(config.model_dir)
        / "language_model"
        / f"{config.model.decoder_num_ngrams}gram.arpa"
    )
    subprocess.run(
        [
            str(kenlm_build_dir / "bin" / "build_binary"),
            str(new_ngram_path),
            str(new_ngram_path.with_suffix(".bin")),
        ]
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
