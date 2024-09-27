"""Training n-gram language model for Wav2Vec2 models."""

import io
import logging
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import requests
from datasets import Dataset, IterableDataset, concatenate_datasets, load_dataset
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pyctcdecode.decoder import build_ctcdecoder
from tqdm.auto import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

from .data import load_dataset_for_evaluation, process_dataset
from .utils import convert_iterable_dataset_to_dataset

logger = logging.getLogger(__package__)


def train_and_store_ngram_model(config: DictConfig) -> None:
    """Trains an n-gram language model and stores it in the model directory.

    Args:
        config:
            Hydra configuration dictionary.
    """
    is_main_process = os.getenv("RANK", "0") == "0"
    if not is_main_process:
        return
    kenlm_build_dir = download_and_compile_kenlm(config=config)
    ngram_model_path = train_ngram_model(kenlm_build_dir=kenlm_build_dir, config=config)
    store_ngram_model(ngram_model_path=ngram_model_path, config=config)
    compress_ngram_model(kenlm_build_dir=kenlm_build_dir, config=config)


def download_and_compile_kenlm(config: DictConfig) -> Path:
    """Download and compile the `kenlm` library.

    Args:
        config:
            Hydra configuration dictionary.

    Returns:
        Path to the `kenlm` build directory.
    """
    # Ensure that the `kenlm` directory exists, and download if otherwise
    cache_dir = (
        Path.home() / ".cache" if config.cache_dir is None else Path(config.cache_dir)
    )

    # Install dependencies if on Ubuntu/Debian
    if shutil.which(cmd="apt-get") is not None:
        logger.info("Installing `kenlm` dependencies...")
        subprocess.run(
            [
                "sudo",
                "apt-get",
                "install",
                "--no-upgrade",
                "-y",
                "build-essential",
                "libboost-all-dev",
                "cmake",
                "libbz2-dev",
                "liblzma-dev",
            ]
        )

    kenlm_dir = cache_dir / "kenlm"
    if not kenlm_dir.exists():
        logger.info("Downloading `kenlm`...")
        with requests.get(url="https://kheafield.com/code/kenlm.tar.gz") as response:
            response.raise_for_status()
            data = response.content
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(path=cache_dir)

    # Compile `kenlm` if it hasn't already been compiled
    kenlm_build_dir = kenlm_dir / "build"
    if not (kenlm_build_dir / "bin" / "lmplz").exists():
        logger.info("Compiling `kenlm`...")
        kenlm_build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmake", ".."], cwd=str(kenlm_build_dir))
        subprocess.run(["make", "-j", "2"], cwd=str(kenlm_build_dir))

    return kenlm_build_dir


def train_ngram_model(kenlm_build_dir: Path, config: DictConfig) -> Path:
    """Train an n-gram language model.

    Args:
        kenlm_build_dir:
            Path to the `kenlm` build directory.
        config:
            Hydra configuration dictionary.

    Returns:
        Path to the trained n-gram language model.
    """
    cache_dir = (
        Path.home() / ".cache" if config.cache_dir is None else Path(config.cache_dir)
    )

    num_ngrams = config.model.decoder_num_ngrams
    correct_ngram_path = Path(config.model_dir) / f"{num_ngrams}gram.arpa"
    if correct_ngram_path.exists():
        return correct_ngram_path

    raw_ngram_path = Path(config.model_dir) / f"raw_{num_ngrams}gram.arpa"
    raw_ngram_path.parent.mkdir(parents=True, exist_ok=True)

    # If the raw language model does not exist either then train from scratch
    if not raw_ngram_path.exists():
        logger.info("Building corpus for training n-gram language model...")
        sentence_path = get_sentence_corpus_path(config=config)

        # Train the n-gram language model
        logger.info("Training n-gram language model...")
        prune_args = ["0"] + ["1"] * (config.model.decoder_num_ngrams - 1)
        with sentence_path.open() as f_in, raw_ngram_path.open("w") as f_out:
            subprocess.run(
                [
                    str(kenlm_build_dir / "bin" / "lmplz"),
                    "-o",  # Order of the n-gram model
                    str(config.model.decoder_num_ngrams),
                    "-S",  # Memory limit
                    "80%",
                    "-T",  # Temporary file location
                    str(cache_dir),
                    "--prune",  # Pruning of the n-gram model
                    *prune_args,
                ],
                stdin=f_in,
                stdout=f_out,
            )

        assert raw_ngram_path.exists(), "Failed to train n-gram language model"

    # Add end-of-sentence marker </s> to the n-gram language model to get the final
    # language model
    logger.info("Adding end-of-sentence marker to n-gram language model...")
    with raw_ngram_path.open("r") as f_in:
        with correct_ngram_path.open("w") as f_out:
            has_added_eos = False
            for line in f_in:
                # Increment the 1-gram count by 1
                if not has_added_eos and "ngram 1=" in line:
                    count = line.strip().split("=")[-1]
                    new_line = line.replace(f"{count}", f"{int(count)+1}")
                    f_out.write(new_line)

                # Add the end-of-sentence marker right after the start-of-sentence marker
                elif not has_added_eos and "<s>" in line:
                    f_out.write(line)
                    f_out.write(line.replace("<s>", "</s>"))
                    has_added_eos = True

                # Otherwise we're just copying the line verbatim
                else:
                    f_out.write(line)

    # Remove non-correct ngram model again
    if raw_ngram_path.exists():
        raw_ngram_path.unlink()

    logger.info(f"Trained n-gram language model stored at {correct_ngram_path}")
    return correct_ngram_path


def get_sentence_corpus_path(config: DictConfig) -> Path:
    """Get the path to the sentence corpus, and create it if it doesn't exist.

    Args:
        config:
            Hydra configuration dictionary.

    Returns:
        Path to the sentence corpus.
    """
    cache_dir = (
        Path.home() / ".cache" if config.cache_dir is None else Path(config.cache_dir)
    )

    dataset_hash = hash(
        tuple([dataset_name for dataset_name in config.decoder_datasets])
    )
    sentence_path = cache_dir / f"ngram-sentences-{dataset_hash}.txt"
    if sentence_path.exists():
        return sentence_path

    all_datasets: list[Dataset] = list()
    for dataset_name, dataset_config in config.decoder_datasets.items():
        logger.info(f"Loading dataset {dataset_name!r}...")

        dataset = load_dataset(
            path=dataset_config.id,
            name=dataset_config.subset,
            split=dataset_config.split,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
            trust_remote_code=True,
            cache_dir=str(cache_dir),
            streaming=True,
        )
        assert isinstance(dataset, IterableDataset)

        if dataset_config.audio_column is not None:
            dataset = dataset.remove_columns(column_names=dataset_config.audio_column)

        if dataset_config.text_column != "text":
            dataset = dataset.rename_column(dataset_config.text_column, "text")

        dataset = convert_iterable_dataset_to_dataset(
            iterable_dataset=dataset, cache_dir=cache_dir
        )
        assert isinstance(dataset, Dataset)

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
        assert isinstance(dataset, Dataset)

        logger.info(
            f"{dataset_name.title().replace('_', ' ')} dataset contains "
            f"{len(dataset):,} examples"
        )

        all_datasets.append(dataset)

    logger.info("Concatenating datasets...")
    dataset = concatenate_datasets(dsets=all_datasets)
    logger.info(f"Concatenated dataset contains {len(dataset):,} examples")

    logger.info("Shuffling dataset...")
    dataset = dataset.shuffle(seed=config.seed)

    # Deduplicating the sentences in the dataset is required when training the n-gram
    # language model
    logger.info("Deduplicating sentences...")
    num_sentences_before = len(dataset["text"])
    sentences = list(set(dataset["text"]))
    logger.info(
        f"Removed {num_sentences_before - len(sentences):,} duplicates from the "
        f"dataset"
    )

    # Load the evaluation sentences, which are not allowed to be in the training dataset
    evaluation_config = DictConfig(
        dict(
            dataset="alexandrainst/coral::read_aloud",
            cache_dir=cache_dir,
            eval_split_name="test",
            text_column="text",
            audio_column="audio",
            sampling_rate=16_000,
            min_seconds_per_example=0.0,
            max_seconds_per_example=1e6,
            clean_text=config.model.clean_text,
            lower_case=config.model.lower_case,
            characters_to_keep="abcdefghijklmnopqrstuvwxyzæøå0123456789éü",
        )
    )
    evaluation_dataset = load_dataset_for_evaluation(config=evaluation_config)
    evaluation_sentences = set(evaluation_dataset[evaluation_config.text_column])

    def remove_evaluation_sentences(sentence: str) -> tuple[str, bool]:
        """Remove evaluation sentences from a sentence.

        Args:
            sentence:
                Sentence to remove evaluation sentences from.

        Returns:
            A tuple containing:
            - The sentence with the evaluation sentences removed.
            - A boolean indicating whether an evaluation sentence was present in
              the sentence.
        """
        evalulation_sentence_present = False
        for evaluation_sentence in evaluation_sentences:
            if evaluation_sentence in sentence:
                sentence = sentence.replace(evaluation_sentence, "")
                evalulation_sentence_present = True
        return sentence, evalulation_sentence_present

    # Remove sentences, that appear in the CoRal test split
    with Parallel(n_jobs=-2) as parallel:
        tuples = parallel(
            delayed(remove_evaluation_sentences)(sentence=sentence)
            for sentence in tqdm(sentences, desc="Removing evaluation sentences")
        )
    sentences = [t[0] for t in tuples if t is not None]
    number_of_sentences_changed = sum(t[1] for t in tuples if t is not None)
    logger.info(
        f"Removed evaluation sentences from {number_of_sentences_changed:,} examples"
    )

    with sentence_path.open("w") as text_file:
        text_file.write("\n".join(sentences))
        text_file.flush()

    return sentence_path


def store_ngram_model(ngram_model_path: Path, config: DictConfig) -> None:
    """Stores the n-gram language model in the model directory.

    Args:
        ngram_model_path:
            Path to the n-gram language model.
        config:
            Hydra configuration dictionary.
    """
    logger.info("Storing n-gram language model...")

    processor = Wav2Vec2Processor.from_pretrained(config.model_dir)

    # Extract the vocabulary, which will be used to build the CTC decoder
    vocab_dict: dict[str, int] = processor.tokenizer.get_vocab()
    sorted_vocab_list = sorted(vocab_dict.items(), key=lambda item: item[1])
    sorted_vocab_dict = {k.lower(): v for k, v in sorted_vocab_list}

    # Build the processor with LM included and save it
    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()), kenlm_model_path=str(ngram_model_path)
    )
    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder,
    )
    processor_with_lm.save_pretrained(config.model_dir)

    # Remove the ngram model again, as the `save_pretrained` method also saves the ngram
    # model
    if ngram_model_path.exists():
        ngram_model_path.unlink()


def compress_ngram_model(kenlm_build_dir: Path, config: DictConfig) -> None:
    """Compress the n-gram language model.

    Args:
        kenlm_build_dir:
            Path to the `kenlm` build directory.
        config:
            Hydra configuration dictionary.
    """
    logger.info("Compressing n-gram language model...")

    compressed_ngram_path = (
        Path(config.model_dir)
        / "language_model"
        / f"{config.model.decoder_num_ngrams}gram.arpa"
    )
    subprocess.run(
        [
            str(kenlm_build_dir / "bin" / "build_binary"),
            str(compressed_ngram_path),
            str(compressed_ngram_path.with_suffix(".bin")),
        ]
    )

    # Remove the uncompressed ngram model, as we only need the compressed version
    if compressed_ngram_path.exists():
        compressed_ngram_path.unlink()
