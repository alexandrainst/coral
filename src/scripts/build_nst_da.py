"""Script that downloads, builds and uploads the Danish part of the NST dataset.

Usage:
    python src/scripts/build_nst_da.py DESTINATION_DIR
"""

import datetime as dt
import logging
import multiprocessing as mp
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
import requests as rq
from datasets import Audio, Dataset, DatasetDict
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_nst_da")


BASE_URL = "https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/dk_2020"


DATA_URLS = dict(
    train_metadata=f"{BASE_URL}/ADB_OD_DAN_0565.tar.gz",
    train_audio=f"{BASE_URL}/lydfiler_16_begge.tar.gz",
    test_and_errors=f"{BASE_URL}/supplement_dk.tar.xz",
    metadata_csvs=f"{BASE_URL}/metadata_dk_csv.zip",
    readme=f"{BASE_URL}/dk-16khz_reorganized_02.pdf",
)


SAMPLE_RATE = 16_000


@click.command("Builds and stores the Danish part of the NST dataset.")
@click.argument("destination_dir", type=click.Path())
def main(destination_dir) -> None:
    """Downloads, builds and uploads the Danish part of the NST dataset.

    Args:
        destination_dir:
            The directory to download and build the dataset in.
    """
    raw_dir = Path(destination_dir) / "raw"
    huggingface_dir = Path(destination_dir) / "huggingface"
    raw_dir.mkdir(parents=True, exist_ok=True)
    huggingface_dir.mkdir(parents=True, exist_ok=True)

    for name, url in DATA_URLS.items():
        filename = name + get_suffix(url)
        destination_path = Path(raw_dir) / filename
        stream_download(url=url, destination_path=destination_path)
        uncompress_file(filename=destination_path)
    reorganise_files(dataset_dir=raw_dir)
    remove_bad_files(dataset_dir=raw_dir)
    dataset = build_huggingface_dataset(dataset_dir=raw_dir)

    logger.info(f"Saving the dataset to {huggingface_dir}...")
    dataset.save_to_disk(
        str(huggingface_dir), max_shard_size="500MB", num_proc=mp.cpu_count() - 1
    )


def stream_download(url: str, destination_path: str | Path) -> None:
    """Download a file from a URL to a destination path.

    Args:
        url:
            The URL to download from.
        destination_path:
            The path to save the file to.
    """
    streamer = rq.get(url, stream=True)
    total_size_in_bytes = int(streamer.headers.get("content-length", 0))
    block_size = 1024
    with Path(destination_path).open(mode="wb") as f:
        pbar = tqdm(
            desc=f"Downloading {url.split('/')[-1]}",
            unit="iB",
            unit_scale=True,
            total=total_size_in_bytes,
        )
        for data in streamer.iter_content(block_size):
            pbar.update(len(data))
            f.write(data)


def uncompress_file(filename: str | Path) -> None:
    """Uncompress a file to a directory.

    Args:
        filename:
            The path to the file to uncompress.
    """
    filename = str(filename)
    match get_suffix(filename):
        case ".tar.gz":
            logger.info(f"Uncompressing {filename}")
            with tarfile.open(filename, mode="r:gz") as tar:
                tar.extractall(path=filename.replace(".tar.gz", ""))
            Path(filename).unlink()
        case ".tar.xz":
            logger.info(f"Uncompressing {filename}")
            with tarfile.open(filename, mode="r:xz") as tar:
                tar.extractall(path=filename.replace(".tar.xz", ""))
            Path(filename).unlink()
        case ".zip":
            logger.info(f"Uncompressing {filename}")
            with zipfile.ZipFile(filename, mode="r") as zip:
                zip.extractall(path=filename.replace(".zip", ""))
            Path(filename).unlink()
        case _:
            pass


def reorganise_files(dataset_dir: str | Path) -> None:
    """Reorganise the files into `train` and `test` directories.

    Args:
        dataset_dir:
            The directory that should contain the dataset.
    """
    logger.info("Reorganising files")

    data_dir = Path(dataset_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_audio_dir = train_dir / "audio"
    test_audio_dir = test_dir / "audio"
    train_audio_dir.mkdir(parents=True, exist_ok=True)
    test_audio_dir.mkdir(parents=True, exist_ok=True)

    for name in DATA_URLS:
        name_dir = data_dir / name
        if not name_dir.exists():
            continue
        match name:
            case "train_metadata":
                name_dir.rename(data_dir / "metadata")
                shutil.move(data_dir / "metadata", train_dir)
            case "train_audio":
                raw_dir = name_dir / "dk"
                for audio_dir in raw_dir.iterdir():
                    if not audio_dir.is_dir():
                        continue
                    for audio_file in audio_dir.glob("*.wav"):
                        Path(audio_file).rename(train_audio_dir / audio_file.name)
                shutil.rmtree(name_dir)

            # This file contains the test set as well as some corrections of errors in
            # the training dataset
            case "test_and_errors":
                raw_dir = name_dir / "supplement_dk"

                temp_test_audio_dir = raw_dir / "testdata" / "audio"
                for audio_dir in temp_test_audio_dir.iterdir():
                    if not audio_dir.is_dir():
                        continue
                    for audio_file in audio_dir.glob("*.wav"):
                        Path(audio_file).rename(test_audio_dir / audio_file.name)
                temp_test_metadata_dir = raw_dir / "testdata" / "metadata"
                shutil.move(temp_test_metadata_dir, test_dir)

                test_log_file = raw_dir / "testdata" / "sprakbanken_0611_transform.log"
                Path(test_log_file).rename(data_dir / "test" / "log.log")

                error_file = raw_dir / "dk_errorfiles_train.json"
                Path(error_file).rename(train_dir / "errorfiles.json")

                test_manifest_file = raw_dir / "testdata" / "dk_manifest_test.json"
                Path(test_manifest_file).rename(test_dir / "manifest.json")

                shutil.rmtree(name_dir)
            case "metadata_csvs":
                train_metadata_csv = name_dir / "NST_dk.csv"
                Path(train_metadata_csv).rename(train_dir / "metadata.csv")
                test_metadata_csv = name_dir / "supplement_dk.csv"
                Path(test_metadata_csv).rename(test_dir / "metadata.csv")
                shutil.rmtree(name_dir)
            case "readme":
                name_dir.rename(data_dir / "README.pdf")


def remove_bad_files(dataset_dir: Path | str) -> None:
    """Remove audio files that cannot be opened.

    Args:
        dataset_dir:
            The directory that should contain the dataset.
    """
    dataset_dir = Path(dataset_dir)

    # These filename prefixes were found by running the `find_faulty_audio_clips.py`
    # script
    bad_file_prefixes = ["dk11x242-18072000-1149_u0047", "dk16xx41-24092000-1951_u0042"]
    for split in ["train", "test"]:
        audio_dir = dataset_dir / split / "audio"
        for audio_file in audio_dir.glob("*.wav"):
            if any(
                [
                    audio_file.stem.startswith(bad_prefix)
                    for bad_prefix in bad_file_prefixes
                ]
            ):
                logger.info(f"Removing {audio_file} as it cannot be opened.")
                audio_file.unlink()
                continue

            # Small files are also bad, as they are likely to be empty
            file_size = audio_file.stat().st_size
            if file_size < 8192:
                logger.info(
                    f"Removing {audio_file.name!r} as its size {file_size} bytes"
                    " is too small."
                )
                audio_file.unlink()


def get_suffix(string: str | Path) -> str:
    """Get the suffix of a string.

    Contrary to Path.suffix, this also works for strings with multiple suffixes.

    Args:
        string:
            The string to get the suffix from.

    Returns:
        The suffix of the string, or an empty string if there is no suffix.
    """
    string = str(string)
    matches = re.search(r"(\.[^.]+)+$", string.split("/")[-1])
    if matches:
        return matches.group()
    else:
        return ""


def build_huggingface_dataset(dataset_dir: Path | str) -> DatasetDict:
    """Sets up the metadata files and builds the Hugging Face dataset.

    Args:
        dataset_dir:
            The directory that should contain the dataset.

    Returns:
        The Hugging Face dataset.
    """
    dataset_dir = Path(dataset_dir)
    rng = np.random.default_rng(seed=4242)

    def ensure_int(value: int | str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def fix_text_column(text: str | float | None) -> str | None:
        if (
            text == "( ... tavshed under denne indspilning ...)"
            or text == ""
            or not isinstance(text, str)
        ):
            return None
        return text

    columns_to_keep = {
        "filename_both_channels": "audio",
        "text": "text",
        "Speaker_ID": "speaker_id",
        "Age": "age",
        "Sex": "sex",
        "Region_of_Dialect": "dialect",
        "RecDate": "recording_date",
        "RecTime": "recording_time",
    }

    dataset_dict: dict[str, Dataset] = dict()
    for split in ["train", "test"]:
        metadata_path = dataset_dir / split / "metadata.csv"
        metadata_df = pd.read_csv(metadata_path, low_memory=False)

        # Keep the desired columns, rename them, ensure that the datatypes are correct
        # and drop rows with missing values
        metadata_df = metadata_df[columns_to_keep.keys()]
        metadata_df = metadata_df.rename(columns=columns_to_keep)
        metadata_df.age = metadata_df.age.map(ensure_int)
        metadata_df.speaker_id = metadata_df.speaker_id.map(ensure_int)
        metadata_df.text = metadata_df.text.map(fix_text_column)
        metadata_df = metadata_df.dropna()
        metadata_df = metadata_df.convert_dtypes()

        # The filenames in the metadata file does not correspond 1-to-1 with the actual
        # names of the audio files, so we extract the audio filename from the
        # information within the metadata filename
        audio_dir = dataset_dir / split / "audio"
        audio_filenames = pd.Series(
            [str(audio_file) for audio_file in audio_dir.glob("*.wav")]
        )
        recording_datetimes: list[str] = list()
        for idx, row in tqdm(
            iterable=metadata_df.iterrows(),
            total=len(metadata_df),
            desc=f"Extracting file names for the {split} split",
        ):
            datetime = dt.datetime.strptime(
                f"{row.recording_date}T{row.recording_time}", "%d %b %YT%H:%M:%S"
            )

            # This version of NST is the "reorganized" version, where the files have
            # been renamed to be independent of the surrounding folder structure.
            # Within these new filenames the old filenames are included, so we extract
            # these
            original_filename = (
                row.audio.lower().split("-")[-1].split("_")[-1].replace(".wav", "")
            )

            # We next get the filename candidates which has the same old filename as
            # well as the same timestamp
            filename_content = datetime.strftime("%H%M") + "[-_]" + original_filename
            filename_candidate_idxs = (
                audio_filenames.str.contains(filename_content).to_numpy().nonzero()[0]
            )
            filename_candidates = audio_filenames[filename_candidate_idxs]

            # If no such filename exists then we set the filename to None, and we will
            # later remove these rows
            if len(filename_candidates) == 0:
                filename = None

            # If there is one candidate then we simply pick that one. If there are
            # multiple candidates then we pick one at random
            else:
                filename = rng.choice(filename_candidates)

            metadata_df.loc[idx, "audio"] = filename
            recording_datetimes.append(datetime.strftime("%Y-%m-%dT%H:%M:%S"))

        metadata_df["recording_datetime"] = recording_datetimes
        metadata_df = metadata_df.dropna()
        metadata_df = metadata_df.drop(columns=["recording_date", "recording_time"])

        # Remove non-existent audio files
        audio_exists = metadata_df.audio.map(lambda path: Path(path).exists())
        metadata_df = metadata_df[audio_exists]

        metadata_df.to_csv(metadata_path, index=False)

        # Build a Hugging Face dataset from the Pandas dataframe
        split_dataset = Dataset.from_pandas(metadata_df, preserve_index=False)
        split_dataset = split_dataset.cast_column(
            "audio", Audio(sampling_rate=SAMPLE_RATE)
        )
        dataset_dict[split] = split_dataset

    return DatasetDict(dataset_dict)


if __name__ == "__main__":
    main()
