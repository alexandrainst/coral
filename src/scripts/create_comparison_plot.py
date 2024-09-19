"""Creates a plot comparing the performance of different models on a dataset."""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("ggplot")


METRIC_NAMES = dict(cer="Character Error Rate", wer="Word Error Rate")


@click.command()
@click.option(
    "--evaluation-file",
    "-f",
    multiple=True,
    type=click.Path(exists=True),
    help="The path to the CSV evaluation file (ending in '-coral-scores', generated "
    "with the `evaluate_model` script.",
    required=True,
)
@click.option(
    "--metric",
    default="cer",
    type=click.Choice(["cer", "wer"]),
    help="The metric to plot.",
)
def main(evaluation_file: tuple[str], metric: str) -> None:
    """Creates a plot comparing the performance of different models on a dataset.

    Args:
        evaluation_file:
            A tuple of paths to the evaluation files, generated with the
            `evaluate_model` script.
        metric:
            The metric to plot. Either "cer" or "wer".
    """
    files = [Path(file) for file in evaluation_file]
    dfs = {
        file.stem[:-13].split("--")[1].replace("oe", "ø"): load_evaluation_df(file=file)
        for file in files
    }
    df = pd.DataFrame.from_records(
        [df[metric].to_dict() for df in dfs.values()],
        index=[name for name in dfs.keys()],
    ).T
    df.plot(
        kind="bar",
        title=f"{METRIC_NAMES[metric.lower()]} by Group (Lower is Better)",
        ylabel=METRIC_NAMES[metric.lower()],
        legend=True,
        figsize=(12, 6),
    )
    plt.tight_layout(pad=2)
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
    age_group_df = (
        raw_df.query("not age_group.isna() and gender.isna() and dialect.isna()")
        .drop(columns=["gender", "dialect"])
        .rename(columns=dict(age_group="group"))
        .reset_index(drop=True)
    )
    gender_df = (
        raw_df.query("not gender.isna() and age_group.isna() and dialect.isna()")
        .drop(columns=["age_group", "dialect"])
        .rename(columns=dict(gender="group"))
        .reset_index(drop=True)
    )
    dialect_df = (
        raw_df.query("not dialect.isna() and age_group.isna() and gender.isna()")
        .drop(columns=["age_group", "gender"])
        .rename(columns=dict(dialect="group"))
        .reset_index(drop=True)
    )
    overall_df = (
        raw_df.query("age_group.isna() and gender.isna() and dialect.isna()")
        .drop(columns=["age_group", "gender", "dialect"])
        .assign(group="overall")
        .reset_index(drop=True)
    )
    df = pd.concat(
        objs=[overall_df, age_group_df, gender_df, dialect_df], ignore_index=True
    ).set_index("group")
    return df


if __name__ == "__main__":
    main()
