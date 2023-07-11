"""Model setup for Wav2Vec 2.0 models."""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Type

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import Repository
from omegaconf import DictConfig
from pyctcdecode.decoder import build_ctcdecoder
from transformers import (
    AutoProcessor,
    BatchEncoding,
    BatchFeature,
    EvalPrediction,
    Trainer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
)
from transformers.data.data_collator import DataCollatorMixin

from .compute_metrics import compute_wer_metrics
from .utils import download_and_extract, transformers_output_ignored

logger = logging.getLogger(__package__)


@dataclass
class DataCollatorCTCWithPadding(DataCollatorMixin):
    """Data collator that will dynamically pad the inputs received.

    Args:
        processor (Wav2Vec2Processor)
            The processor used for proccessing the data.
        padding (bool, str or PaddingStrategy, optional):
            Select a strategy to pad the returned sequences (according to the model's
            padding side and padding index) among:
            * True or 'longest':
                Pad to the longest sequence in the batch (or no padding if only a
                single sequence if provided).
            * 'max_length':
                Pad to a maximum length specified with the argument max_length or to
                the maximum acceptable input length for the model if that argument is
                not provided.
            * False or 'do_not_pad':
                No padding (i.e., can output a batch with sequences of different
                lengths).
            Defaults to True.
    """

    processor: Wav2Vec2Processor
    padding: bool | str = True
    return_tensors: str = "pt"

    def torch_call(self, features: list[dict]) -> BatchFeature:
        """Collate the features.

        Args:
            features (list of dict):
                A list of feature dicts.

        Returns:
            BatchFeature:
                A dictionary of the collated features.
        """
        # Split inputs and labels since they have to be of different lengths and need
        # different padding methods
        audio_features = [
            {
                "input_values": self.processor(
                    feature["audio"]["array"],
                    sampling_rate=feature["audio"]["sampling_rate"],
                ).input_values[0]
            }
            for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch: BatchFeature = self.processor.pad(
            audio_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch: BatchEncoding = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace padding with -100 to ignore loss correctly
        non_one_entries: torch.Tensor = labels_batch.attention_mask.ne(1)
        labels: torch.Tensor = labels_batch.input_ids.masked_fill(non_one_entries, -100)

        batch["labels"] = labels
        return batch


class Wav2Vec2ModelSetup:
    """Model setup for Wav2Vec 2.0 models.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.processor: Wav2Vec2Processor

    def load_processor(self) -> Wav2Vec2Processor:
        # We dump the vocabulary to a file since the tokenizer uses this file during
        # initialisation
        dump_vocabulary(self.cfg)
        tokenizer: Wav2Vec2CTCTokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            self.cfg.model_dir,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            word_delimiter_token="|",
        )

        # Set the `model_max_length` attribute of the tokenizer, if it hasn't been set,
        # to ensure that truncation is done correctly
        if tokenizer.model_max_length is None or tokenizer.model_max_length > 1e6:
            tokenizer.model_max_length = 512

        extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.cfg.model.sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor=extractor, tokenizer=tokenizer
        )
        return self.processor

    def load_model(self) -> Wav2Vec2ForCTC:
        with transformers_output_ignored():
            logger.info("Initialising model...")
            model = Wav2Vec2ForCTC.from_pretrained(
                self.cfg.model.pretrained_model_id,
                activation_dropout=self.cfg.model.activation_dropout,
                attention_dropout=self.cfg.model.attention_dropout,
                hidden_dropout=self.cfg.model.hidden_dropout,
                feat_proj_dropout=self.cfg.model.feat_proj_dropout,
                final_dropout=self.cfg.model.final_dropout,
                mask_time_prob=self.cfg.model.mask_time_prob,
                mask_feature_prob=self.cfg.model.mask_feature_prob,
                mask_feature_length=self.cfg.model.mask_feature_length,
                layerdrop=self.cfg.model.layerdrop,
                ctc_loss_reduction=self.cfg.model.ctc_loss_reduction,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                vocab_size=len(self.processor.tokenizer.get_vocab()),
            )
            assert isinstance(model, Wav2Vec2ForCTC)

        if self.cfg.model.freeze_feature_encoder:
            logger.debug("Freezing feature encoder...")
            for param in model.wav2vec2.parameters():
                param.requires_grad = False

        return model

    def load_data_collator(self) -> DataCollatorCTCWithPadding:
        return DataCollatorCTCWithPadding(processor=self.processor, padding=True)

    def load_trainer_class(self) -> Type[Trainer]:
        return Trainer

    def load_compute_metrics(self) -> Callable[[EvalPrediction], dict]:
        return partial(compute_wer_metrics, processor=self.processor)


def train_ngram_model(cfg: DictConfig) -> None:
    """Trains an ngram language model.

    Args:
        cfg (DictConfig):
            Hydra configuration dictionary.
    """
    # Load the dataset
    dataset = load_dataset(
        path=cfg.model.decoder.dataset_id,
        name=cfg.model.decoder.dataset_subset,
        split=cfg.model.decoder.dataset_split,
        use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )
    assert isinstance(dataset, Dataset)

    # Ensure that the `kenlm` directory exists, and download if otherwise
    kenlm_dir = Path.home() / ".cache" / "kenlm"
    if not kenlm_dir.exists():
        download_and_extract(
            url="https://kheafield.com/code/kenlm.tar.gz",
            target_dir=kenlm_dir,
        )

    # Compile `kenlm` if it hasn't already been compiled
    kenlm_build_dir = kenlm_dir / "build"
    if not kenlm_build_dir.exists():
        kenlm_build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmake", ".."], cwd=str(kenlm_dir / "build"))
        subprocess.run(["make", "-j", "2"], cwd=str(kenlm_dir / "build"))

    # Train the n-gram language model if it doesn't already exist
    correct_ngram_path = Path(cfg.model_dir) / f"{cfg.model.decoder.n}gram.arpa"
    if not correct_ngram_path.exists():
        ngram_path = Path(cfg.model_dir) / f"raw_{cfg.model.decoder.n}gram.arpa"

        # If the raw language model does not exist either then train from scratch
        if not ngram_path.exists():
            # Dump dataset to a temporary text file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as text_file:
                with text_file.open() as f_in:
                    f_in.write(" ".join(dataset["text"]))
                    with ngram_path.open("w") as f_out:
                        subprocess.run(
                            ["kenlm/build/bin/lmplz", "-o", cfg.model.decoder.n],
                            stdin=f_in,
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

    # Load the pretrained processor
    processor = AutoProcessor.from_pretrained(cfg.model_dir)

    # Extract the vocabulary, which will be used to build the CTC decoder
    vocab_dict: dict[str, int] = processor.tokenizer.get_vocab()
    sorted_vocab_list = sorted(vocab_dict.items(), key=lambda item: item[1])
    sorted_vocab_dict = {k.lower(): v for k, v in sorted_vocab_list}

    # Build the CTC decoder
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
    repo = Repository(local_dir=cfg.model_dir, clone_from=cfg.model_dir)

    # Save the new processor to the repo
    processor_with_lm.save_pretrained(cfg.model_dir)

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


def dump_vocabulary(cfg: DictConfig) -> None:
    """Extracts the vocabulary from the dataset and dumps it to a file.

    It will dump the file to `${cfg.model_dir}/vocab.json`.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
    """
    # Build the set of all unique characters in the dataset
    unique_characters: set[str] = set(cfg.characters_to_keep)

    # Build vocabulary
    vocab = {char: idx for idx, char in enumerate(unique_characters)}
    for tok in ["<unk>", "<pad>", "<s>", "</s>"]:
        vocab[tok] = len(vocab)

    # Dump the vocabulary to a json file
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = model_dir / "vocab.json"
    with vocab_path.open("w") as f:
        json.dump(vocab, f)
