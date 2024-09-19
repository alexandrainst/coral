"""Script that downloads and builds the Nota dataset.

Usage:
    python src/scripts/build_nota.py DESTINATION_DIR
"""

import logging
import multiprocessing as mp
import re
from pathlib import Path
from time import sleep
from urllib.error import ContentTooShortError
from zipfile import ZipFile

import click
import pandas as pd
import requests
from datasets import Audio, Dataset, DatasetDict
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_nota")


BASE_URL = "https://sprogtek-ressources.digst.govcloud.dk/nota/"


URL_SUFFIXES = [
    "Inspiration%202016%20-%202021/",
    "Inspiration%202008%20-%202016/",
    "Radio-TV%20program%202007%20-%202012/",
    "Radio-TV%20Program%202013%20-%202015/",
    "Radio-TV%20Program%202016%20-%202018/",
    "Radio-TV%20Program%202019%20-%202022/",
]


SAMPLE_RATE = 16_000


@click.command("Builds and stores the Nota dataset.")
@click.argument("destination_dir", type=click.Path())
def main(destination_dir) -> None:
    """Downloads and builds the Nota dataset.

    Args:
        destination_dir:
            The directory to download and build the dataset in.
    """
    raw_dir = Path(destination_dir) / "raw"
    huggingface_dir = Path(destination_dir) / "huggingface"
    download_nota(destination_dir=raw_dir)
    dataset = build_huggingface_dataset(dataset_dir=raw_dir)

    logger.info(f"Saving the dataset to {huggingface_dir}...")
    dataset.save_to_disk(
        str(huggingface_dir), max_shard_size="500MB", num_proc=mp.cpu_count() - 1
    )


def download_nota(destination_dir: Path | str) -> None:
    """Downloads the Nota dataset.

    Args:
        destination_dir:
            The directory to download the dataset to.
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    outer_pbar_desc = f"Downloading Nota files to the '{destination_dir}' directory"
    for url_suffix in tqdm(URL_SUFFIXES, desc=outer_pbar_desc):
        # Define the URL to download the dataset from
        url = BASE_URL + url_suffix

        # Get URLs to all files on the URL
        regex = re.compile(r"<a href=\"(.+?)\">")
        all_files: list[str] = regex.findall(str(requests.get(url).content))

        # Ignore the Parent and README files
        all_files_filtered = list(
            filter(lambda x: x != "Readme.txt" and x != "/nota/", all_files)
        )

        # Remove empty files from the list
        empty_files = {"INSL20210003.zip"}
        all_files_filtered = list(set(all_files_filtered) - empty_files)

        # Remove already downloaded files
        downloaded_files = {
            f"{path.stem}.zip"
            for path in Path(destination_dir).iterdir()
            if path.is_dir()
        }
        all_files_filtered = list(set(all_files_filtered) - downloaded_files)

        # Iterate over the files and download them
        inner_pbar_desc = f"Downloading files from {url}"
        for file in tqdm(all_files_filtered, leave=False, desc=inner_pbar_desc):
            file_url: str = (url + file).replace("%20", " ")
            while True:
                try:
                    destination_path = Path(destination_dir) / file
                    stream_download(url=file_url, destination_path=destination_path)
                    break
                except ContentTooShortError:
                    sleep(5)
                    continue

            # Extract the dataset
            with ZipFile(destination_path, "r") as zip_file:
                zip_file.extractall(path=destination_dir)
            Path(destination_path).unlink()


def build_huggingface_dataset(dataset_dir: Path | str) -> DatasetDict:
    """Builds the HuggingFace dataset.

    Args:
        dataset_dir:
            The directory to build the dataset from.

    Returns:
        The Hugging Face dataset.
    """
    dataset_dir = Path(dataset_dir)

    logger.info("Building the HuggingFace dataset...")

    # Build metadata file for the dataset
    metadata_dict: dict[str, list[str]] = dict(audio=[], text=[])
    for folder in dataset_dir.iterdir():
        if not folder.is_dir():
            continue
        for text_file in folder.glob("*.txt"):
            wav_filename = text_file.stem + ".wav"
            metadata_dict["audio"].append(str(folder / wav_filename))
            metadata_dict["text"].append(text_file.read_text(encoding="utf-8"))
    metadata_df = pd.DataFrame(metadata_dict)

    # Build the dataset
    dataset = Dataset.from_pandas(metadata_df, preserve_index=False)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    return DatasetDict(dict(train=dataset))


def stream_download(url: str, destination_path: str | Path) -> None:
    """Download a file from a URL to a destination path.

    Args:
        url:
            The URL to download from.
        destination_path:
            The path to save the file to.
    """
    streamer = requests.get(url, stream=True)
    total_size_in_bytes = int(streamer.headers.get("content-length", 0))
    block_size = 1024
    with Path(destination_path).open(mode="wb") as f:
        pbar = tqdm(
            desc=f"Downloading {url.split('/')[-1]}",
            unit="iB",
            unit_scale=True,
            total=total_size_in_bytes,
            leave=False,
        )
        for data in streamer.iter_content(block_size):
            pbar.update(len(data))
            f.write(data)


if __name__ == "__main__":
    main()
