"""Script that builds the FTSpeech dataset from the raw data.

Usage:
    build_ftspeech.py <raw_data_dir> <output_dir>
"""

import click

from coral_models.asr.ftspeech import build_and_store_data


@click.command()
@click.argument("raw_data_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--n-jobs", "-n", type=int, default=-1)
def main(raw_data_dir: str, output_dir: str, n_jobs: int) -> None:
    """Builds and stores the FTSpeech dataset.

    Args:
        raw_data_dir (str):
            The directory where the raw dataset is stored.
        output_dir (str):
            The path to the resulting dataset.
        n_jobs (int, optional):
            The number of jobs to use for parallel processing. Can be a negative number
            to use all available cores minus `n_jobs`. Defaults to -1, meaning all
            available cores minus 1.
    """
    build_and_store_data(input_dir=raw_data_dir, output_dir=output_dir, n_jobs=n_jobs)


if __name__ == "__main__":
    main()
