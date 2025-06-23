"""Creates a plot comparing the performance of different models on a dataset.

Usage:
    python src/scripts/create_comparison_plot.py \
        -folder EVALUATION_FILE \
        [-f EVALUATION_FILE ...] \
        [--metric METRIC]
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("ggplot")


METRIC_NAMES = dict(cer="Character Error Rate", wer="Word Error Rate")


@click.command()
@click.option(
    "--folder",
    "-d",
    type=click.Path(exists=True, file_okay=False),
    help="The folder containing CSV evaluation files.",
    required=True,
)
def main(folder: str) -> None:
    """Creates plots comparing the performance of different models on a dataset.

    Args:
        folder:
            The path to the folder containing the evaluation files.
    """
    folder_path = Path(folder)
    files = list(folder_path.glob("*-scores.csv"))

    if not files:
        print("No evaluation files found in the specified folder.")
        return

    dataset_name = folder_path.stem  # Use folder name as dataset name

    dfs = {
        file.stem[:-7].split("--")[1].replace("oe", "ø"): load_evaluation_df(file=file)
        for file in files
    }

    for metric in ["cer", "wer"]:
        df = pd.DataFrame.from_records(
            [df[metric].to_dict() for df in dfs.values()],
            index=[name for name in dfs.keys()],
        ).T
        df = df.reindex(sorted(df.columns), axis=1)

        df.plot(
            kind="bar",
            title=f"{METRIC_NAMES[metric.lower()]} on {dataset_name} (Lower is Better)",
            ylabel=METRIC_NAMES[metric.lower()],
            legend=True,
            figsize=(12, 6),
            rot=25,
        )
        plt.tight_layout(pad=2)
        plt.savefig(folder_path / f"{dataset_name}_{metric}_results_vis.png")
        plt.show()


def load_evaluation_df(file: Path) -> pd.DataFrame:
    """Loads the evaluation data from a CSV file into a pandas DataFrame.

    Args:
        file:
            The path to the evaluation file.

    Returns:
        A pandas DataFrame containing the evaluation data.
    """
    raw_df = pd.read_csv(file, index_col=None)
    gender_df = (
        raw_df.query("not gender.isna() and age.isna() and dialect.isna()")
        .drop(columns=["age", "dialect"])
        .rename(columns=dict(gender="group"))
        .reset_index(drop=True)
    ).sort_values(by="group")
    age_df = (
        raw_df.query("not age.isna() and gender.isna() and dialect.isna()")
        .drop(columns=["gender", "dialect"])
        .rename(columns=dict(age="group"))
        .reset_index(drop=True)
    ).sort_values(by="group")
    dialect_df = (
        raw_df.query(
            "not (dialect.isna() or dialect == 'Non-native') "
            "and age.isna() and gender.isna()"
        )
        .drop(columns=["age", "gender"])
        .rename(columns=dict(dialect="group"))
        .reset_index(drop=True)
    ).sort_values(by="group")
    accent_df = (
        raw_df.query("dialect == 'Non-native' and age.isna() and gender.isna()")
        .drop(columns=["age", "gender"])
        .rename(columns=dict(dialect="group"))
        .reset_index(drop=True)
    ).sort_values(by="group")
    overall_df = (
        raw_df.query("age.isna() and gender.isna() and dialect.isna()")
        .drop(columns=["age", "gender", "dialect"])
        .assign(group="overall")
        .reset_index(drop=True)
    ).sort_values(by="group")
    df = (
        pd.concat(
            objs=[gender_df, age_df, dialect_df, accent_df, overall_df],
            ignore_index=True,
        )
        .map(lambda x: x.lower() if isinstance(x, str) else x)
        .set_index("group")
    )
    return df


if __name__ == "__main__":
    main()
