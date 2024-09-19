"""Script that downloads the FTSpeech corpus.

Usage:
    python src/scripts/download_ftspeech.py DOWNLOAD_DIR
"""

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options
except ImportError:
    raise ImportError(
        "To run this script, `selenium` needs to be installed. Please install it with "
        "`pip install selenium` or `poetry add selenium`."
    )

import logging
import time
from getpass import getpass
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_ftspeech")


@click.command("Downloads the FTSpeech corpus.")
@click.argument("download_dir", type=click.Path())
def main(download_dir) -> None:
    """Downloads the FTSpeech corpus.

    Args:
        download_dir:
            The directory to download the FTSpeech corpus to.
    """
    download_dir = Path(download_dir)

    options = Options()
    options.add_argument("--headless")
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.download.dir", str(download_dir))
    options.set_preference(
        "browser.helperApps.neverAsk.saveToDisk", "application/x-gzip"
    )
    driver = webdriver.Firefox(options=options)

    # Go to the FTSpeech download page
    driver.get("http://130.226.140.126")

    # Get the access token input field
    access_token_input = driver.find_element(By.XPATH, '//input[@name="token"]')

    # Get the access token
    logger.info(
        "Please enter your access token, which can be obtained by filling in the form "
        "at https://docs.google.com/forms/d/e/1FAIpQLSdv0OweHtKh__u0J0M8cjtm9hg3Yo_y8b"
        "WxVjQLLUeEyCAxjw/viewform."
    )
    access_token = getpass("Access token: ")

    # Enter the access token
    access_token_input.send_keys(access_token)

    # Get the download button
    download_button = driver.find_element(By.XPATH, '//input[@type="submit"]')

    # Click the download button
    download_button.click()

    # Wait for the download to finish
    while len(list(download_dir.glob("*.part"))) > 0:
        # Get the size of the part file
        part_file_bytes = list(download_dir.glob("*.part"))[0].stat().st_size

        # Convert the bytes to gigabytes
        part_file_gbs = part_file_bytes / 1024 / 1024 / 1024

        # Log the download progress
        print(f"Downloading FTSpeech - {part_file_gbs:.2f} GB...", end="\r")

        # Wait for a second
        time.sleep(1)

    # Wait for an hour, to ensure that it has finished downloading
    time.sleep(60 * 60)

    # Close the webdriver
    driver.close()


if __name__ == "__main__":
    main()
