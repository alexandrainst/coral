"""General utility functions."""

import io
import tarfile
import warnings
from pathlib import Path

import datasets.utils.logging as ds_logging
import requests
import transformers.utils.logging as hf_logging
from datasets.utils import disable_progress_bar


def block_terminal_output() -> None:
    """Blocks undesired terminal output.

    This blocks the following output:
        - User warnings
        - Logs from the `datasets` package
        - Progress bars from the `datasets` package
    """

    # Ignore user warnings throughout the codebase
    warnings.filterwarnings("ignore", category=UserWarning)

    # Disable logging from the `datasets` library
    ds_logging.set_verbosity_error()

    # Disable the tokeniser progress bars from the `datasets` library
    disable_progress_bar()


class transformers_output_ignored:
    """Context manager to block terminal output."""

    def __enter__(self) -> None:
        hf_logging.set_verbosity_error()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        hf_logging.set_verbosity_info()


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
