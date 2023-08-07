"""Script that downloads, builds and uploads the Danish part of the NST dataset.

Usage:
    python build_nst_da.py <destination_dir>
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
import pandas as pd
import requests as rq
from datasets import Audio, Dataset, DatasetDict
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


BASE_URL = "https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/dk_2020"


DATA_URLS = dict(
    train_metadata=f"{BASE_URL}/ADB_OD_DAN_0565.tar.gz",
    train_audio=f"{BASE_URL}/lydfiler_16_begge.tar.gz",
    test_and_errors=f"{BASE_URL}/supplement_dk.tar.xz",
    metadata_csvs=f"{BASE_URL}/metadata_dk_csv.zip",
    readme=f"{BASE_URL}/dk-16khz_reorganized_02.pdf",
)


@click.command("Builds and stores the Danish part of the NST dataset.")
@click.argument("output_dir", type=click.Path())
def main(output_dir) -> None:
    for name, url in DATA_URLS.items():
        filename = name + get_suffix(url)
        destination_path = Path(output_dir) / filename
        stream_download(url=url, destination_path=destination_path)
        uncompress_file(filename=destination_path)
    reorganise_files(dataset_dir=output_dir)
    dataset = build_huggingface_dataset()

    logger.info(f"Saving the dataset to {output_dir}...")
    dataset.save_to_disk(
        str(output_dir),
        max_shard_size="50MB",
        num_proc=mp.cpu_count() - 1,
    )


def stream_download(url: str, destination_path: str | Path) -> None:
    """Download a file from a URL to a destination path.

    Args:
        url: The URL to download from.
        destination_path: The path to save the file to.
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
        filename: The path to the file to uncompress.
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
        dataset_dir: The directory that should contain the dataset.
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
        if not Path(name).exists():
            continue
        match name:
            case "train_metadata":
                Path(name).rename("metadata")
                shutil.move("metadata", train_dir)
            case "train_audio":
                data_dir = Path(name) / "dk"
                for audio_dir in data_dir.iterdir():
                    if not audio_dir.is_dir():
                        continue
                    for audio_file in audio_dir.glob("*.wav"):
                        Path(audio_file).rename(train_audio_dir / audio_file.name)
                shutil.rmtree(name)
            case "test_and_errors":
                data_dir = Path(name) / "supplement_dk"

                temp_test_audio_dir = data_dir / "testdata" / "audio"
                for audio_dir in temp_test_audio_dir.iterdir():
                    if not audio_dir.is_dir():
                        continue
                    for audio_file in audio_dir.glob("*.wav"):
                        Path(audio_file).rename(test_audio_dir / audio_file.name)
                temp_test_metadata_dir = data_dir / "testdata" / "metadata"
                shutil.move(temp_test_metadata_dir, test_dir)

                test_log_file = data_dir / "testdata" / "sprakbanken_0611_transform.log"
                Path(test_log_file).rename(data_dir / "test" / "log.log")

                error_file = data_dir / "dk_errorfiles_train.json"
                Path(error_file).rename(train_dir / "errorfiles.json")

                test_manifest_file = data_dir / "dk_manifest_test.json"
                Path(test_manifest_file).rename(test_dir / "manifest.json")

                shutil.rmtree(name)
            case "metadata_csvs":
                train_metadata_csv = Path(name) / "NST_dk.csv"
                Path(train_metadata_csv).rename(train_dir / "metadata.csv")
                test_metadata_csv = Path(name) / "supplement_dk.csv"
                Path(test_metadata_csv).rename(test_dir / "metadata.csv")
                shutil.rmtree(name)
            case "readme":
                Path(name).rename("README.pdf")


def get_suffix(string: str | Path) -> str:
    """Get the suffix of a string.

    Contrary to Path.suffix, this also works for strings with multiple suffixes.

    Args:
        string: The string to get the suffix from.

    Returns:
        The suffix of the string, or an empty string if there is no suffix.
    """
    string = str(string)
    matches = re.search(r"(\.[^.]+)+$", string.split("/")[-1])
    if matches:
        return matches.group()
    else:
        return ""


def build_huggingface_dataset() -> DatasetDict:
    """Sets up the metadata files and builds the Hugging Face dataset.

    Returns:
        The Hugging Face dataset.
    """

    def ensure_int(value: int | str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def fix_text_column(text: str) -> str:
        if text == "( ... tavshed under denne indspilning ...)":
            return ""
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
        metadata_path = Path(split) / "metadata.csv"
        metadata_df = pd.read_csv(metadata_path, low_memory=False)
        metadata_df = metadata_df[columns_to_keep.keys()]
        metadata_df = metadata_df.rename(columns=columns_to_keep)
        metadata_df.age = metadata_df.age.map(ensure_int)
        metadata_df.speaker_id = metadata_df.age.map(ensure_int)
        metadata_df.text = metadata_df.text.map(fix_text_column)
        metadata_df = metadata_df.dropna()
        metadata_df = metadata_df.convert_dtypes()

        audio_dir = Path(split) / "audio"
        recording_datetimes: list[str] = list()
        for idx, row in tqdm(
            iterable=metadata_df.iterrows(),
            total=len(metadata_df),
            desc=f"Extracting file names for the {split} split",
        ):
            datetime = dt.datetime.strptime(
                f"{row.recording_date}T{row.recording_time}", "%d %b %YT%H:%M:%S"
            )
            original_filename = (
                row.audio.lower().split("-")[-1].split("_")[-1].replace(".wav", "")
            )
            file_name_content = datetime.strftime("%H%M") + "[-_]" + original_filename
            file_name_candidates = sorted(
                audio_dir.glob(f"*x{row.speaker_id:02d}*{file_name_content}*.wav")
            )
            if len(file_name_candidates) == 0:
                file_name = None
            else:
                file_name = str(file_name_candidates[0])
            metadata_df.loc[idx, "audio"] = file_name
            recording_datetimes.append(datetime.strftime("%Y-%m-%dT%H:%M:%S"))

        metadata_df["recording_datetime"] = recording_datetimes
        metadata_df = metadata_df.drop(columns=["recording_date", "recording_time"])

        metadata_df.to_csv(metadata_path, index=False)

        split_dataset = Dataset.from_pandas(metadata_df, preserve_index=False)
        split_dataset = split_dataset.cast_column("audio", Audio(sampling_rate=16_000))
        dataset_dict[split] = split_dataset

    return DatasetDict(dataset_dict)


if __name__ == "__main__":
    main()
