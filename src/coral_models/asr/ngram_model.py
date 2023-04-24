"""Language model to boost performance of the speech recognition model."""

import os
from pathlib import Path
from shutil import rmtree

from datasets import Dataset, load_dataset
from huggingface_hub import Repository
from pyctcdecode.decoder import build_ctcdecoder
from transformers import AutoProcessor, Wav2Vec2ProcessorWithLM


def train_ngram_model(
    model_id: str,
    dataset_id: str = "DDSC/reddit-da-asr-preprocessed",
    split: str = "train",
    n: int = 5,
) -> None:
    """Trains an ngram language model.

    Args:
        model_id (str):
            The model id of the finetuned speech model, which we will merge
            with the ngram model.
        dataset_id (str, optional):
            The dataset to use for training. Defaults to
            'DDSC/reddit-da-asr-preprocessed'.
        split (str, optional):
            The split to use for training. Defaults to 'train'.
        n (int, optional):
            The ngram order to use for training. Defaults to 5.
    """
    # Ensure that the data folder exists
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()

    # Ensure that the models folder exists
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir()

    # Load the dataset
    try:
        dataset = load_dataset(dataset_id, split=split, use_auth_token=True)
    except ValueError:
        dataset = Dataset.from_file(f"{dataset_id}/dataset.arrow")

    # Dump dataset to a text file
    text_path = data_dir / "text_data.txt"
    with open(text_path, "w") as f:
        f.write(" ".join(dataset["text"]))

    # Ensure that the `kenlm` directory exists, and download if otherwise
    kenlm_dir = Path("kenlm")
    if not kenlm_dir.exists():
        os.system("wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz")

    # Compile `kenlm` if it hasn't already been compiled
    kenlm_build_dir = kenlm_dir / "build"
    if not kenlm_build_dir.exists():
        os.system(
            "mkdir kenlm/build && " "cd kenlm/build && " "cmake .. && " "make -j2"
        )

    # Train the n-gram language model if it doesn't already exist
    correct_ngram_path = models_dir / f"{n}gram.arpa"
    if not correct_ngram_path.exists():
        # If the raw language model does not exist either, then train from
        # scratch
        ngram_path = models_dir / f"raw_{n}gram.arpa"
        if not ngram_path.exists():
            os.system(
                f"kenlm/build/bin/lmplz -o {n} <" f'"{text_path}" >' f'"{ngram_path}"'
            )

        # Add end-of-sentence marker </s> to the n-gram language model to get
        # the final language model
        with ngram_path.open("r") as f_in:
            with correct_ngram_path.open("w") as f_out:
                # Iterate over the lines in the input file
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
    processor = AutoProcessor.from_pretrained(model_id, use_auth_token=True)

    # Extract the vocabulary, which will be used to build the CTC decoder
    vocab_dict = processor.tokenizer.get_vocab()
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
    repo_dir = models_dir / model_id.split("/")[-1]
    repo = Repository(local_dir=str(repo_dir), clone_from=model_id)

    # Remove the previous language model if it exists
    lang_model_dir = repo_dir / "language_model"
    if lang_model_dir.exists():
        rmtree(str(lang_model_dir))

    # Save the new processor to the repo
    processor_with_lm.save_pretrained(str(repo_dir))

    # Compress the ngram model
    os.system(
        f"kenlm/build/bin/build_binary "
        f"{repo_dir}/language_model/{n}gram.arpa "
        f"{repo_dir}/language_model/{n}gram.bin"
    )

    # Remove the uncompressed ngram model
    uncompressed_path = repo_dir / "language_model" / f"{n}gram.arpa"
    if uncompressed_path.exists():
        uncompressed_path.unlink()

    # Push the changes to the repo
    repo.push_to_hub(commit_message="Upload LM-boosted decoder")


if __name__ == "__main__":
    model_ids = [
        "saattrupdan/alvenir-wav2vec2-base-cv8-da",
    ]
    for model_id in model_ids:
        train_ngram_model(model_id, dataset_id="data/lexdk-preprocessed")
