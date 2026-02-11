"""Creates a plot comparing the performance of different models on a dataset.

Usage:
    python src/scripts/create_comparison_plot.py \
        -f EVALUATION_FILE \
        [-f EVALUATION_FILE ...] \
        [--metric METRIC]
"""

import itertools as it
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("ggplot")


METRIC_NAMES = dict(cer="Character error rate", wer="Word error rate")


@click.command()
@click.option(
    "--evaluation-file",
    "-f",
    multiple=True,
    type=click.Path(path_type=Path),
    help="The path to the CSV evaluation file (generated with the `evaluate_model` "
    "script). Can be specified multiple times to compare multiple models. Can also "
    "specify a glob pattern to compare all matching files.",
    required=True,
)
@click.option(
    "--metric",
    default="cer",
    type=click.Choice(["cer", "wer"]),
    help="The metric to plot.",
)
@click.option(
    "--model-focus",
    "-m",
    multiple=True,
    type=str,
    help="If specified, mark these models in a separate colour for emphasis.",
)
@click.option("--title", default=None, type=str, help="The title of the plot.")
def main(
    evaluation_file: tuple[Path, ...],
    metric: str,
    model_focus: tuple[str],
    title: str | None,
) -> None:
    """Creates a plot comparing the performance of different models on a dataset.

    Raises:
        ValueError:
            If the metric is not supported.
    """
    # Glob evaluation files if needed
    glob_files = [
        file if "*.csv" in file.name else Path(file.as_posix().replace("*", "*.csv"))
        for file in evaluation_file
        if "*" in file.name
    ]
    if glob_files:
        evaluation_file = tuple(
            [
                file
                for glob_file in glob_files
                for file in glob_file.parent.glob(glob_file.name)
            ]
            + [file for file in evaluation_file if "*" not in file.name]
        )

    metric_name = METRIC_NAMES[metric.lower()]
    dataset_names = {
        file.stem.split(".")[1].split("--")[-1] for file in evaluation_file
    }

    dfs: dict[str, pd.DataFrame] = dict()
    column_order: list[str] = []
    for dataset_name in dataset_names:
        sub_dfs = {
            file.stem.split(".")[0]
            .replace("oe", "ø")
            .replace("ae", "æ"): load_evaluation_df(file=file)
            for file in evaluation_file
            if file.stem.split(".")[1].split("--")[-1] == dataset_name
        }
        df = pd.DataFrame.from_records(
            [df[metric].to_dict() for df in sub_dfs.values()],
            index=[name for name in sub_dfs.keys()],
        ).sort_index()
        dfs[dataset_name] = df
        column_order = df.columns.tolist()

    # Check that each dataset has the same models
    model_sets = [set(df.index) for df in dfs.values()]
    if not all(model_set == model_sets[0] for model_set in model_sets[1:]):
        raise ValueError(
            "The evaluation files must contain the same models for each dataset."
        )

    # Merge dataframes by averaging the metrics across datasets
    df = sum(dfs.values()) / len(dfs)
    df.dropna(axis=1, how="any", inplace=True)
    df.sort_values(by="overall", ascending=False, inplace=True)
    df = df.loc[:, [col for col in column_order if col in df.columns]]

    # Sort colours by model name, unless model focus is specified, in which case
    # we colour the focused models in shades of the same colour and the rest in shades
    # of a more faded colour
    if model_focus:
        red_colours = it.cycle(plt.cm.tab20c.colors[4:8])
        grey_colours = it.cycle(plt.cm.tab20c.colors[16:20])
        model_colours = {
            model: next(red_colours) if model in model_focus else next(grey_colours)
            for model in sorted(df.index)
        }
        colours = [model_colours[model] for model in df.index]
    else:
        colours = [
            colour
            for _, colour in sorted(
                zip(df.index, plt.cm.Set3.colors[: df.shape[0]]), key=lambda x: x[0]
            )
        ]

    df.T.plot(
        kind="bar",
        title=(
            f"{metric_name} by group on {dataset_name} (lower is better)"
            if title is None
            else title
        ),
        ylabel=METRIC_NAMES[metric.lower()],
        legend=True,
        figsize=(12, 6),
        rot=25,
        color=colours,
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
    gender_df = (
        raw_df.query("not gender.isna() and age_group.isna() and dialect.isna()")
        .drop(columns=["age_group", "dialect"])
        .rename(columns=dict(gender="group"))
        .reset_index(drop=True)
    ).sort_values(by="group")
    age_group_df = (
        raw_df.query("not age_group.isna() and gender.isna() and dialect.isna()")
        .drop(columns=["gender", "dialect"])
        .rename(columns=dict(age_group="group"))
        .reset_index(drop=True)
    ).sort_values(by="group")
    dialect_df = (
        raw_df.query(
            "not (dialect.isna() or dialect == 'Non-native') "
            "and age_group.isna() and gender.isna()"
        )
        .drop(columns=["age_group", "gender"])
        .rename(columns=dict(dialect="group"))
        .reset_index(drop=True)
    ).sort_values(by="group")
    accent_df = (
        raw_df.query("dialect == 'Non-native' and age_group.isna() and gender.isna()")
        .drop(columns=["age_group", "gender"])
        .rename(columns=dict(dialect="group"))
        .reset_index(drop=True)
    ).sort_values(by="group")
    overall_df = (
        raw_df.query("age_group.isna() and gender.isna() and dialect.isna()")
        .drop(columns=["age_group", "gender", "dialect"])
        .assign(group="overall")
        .reset_index(drop=True)
    ).sort_values(by="group")
    df = (
        pd.concat(
            objs=[gender_df, age_group_df, dialect_df, accent_df, overall_df],
            ignore_index=True,
        )
        .map(lambda x: x.lower() if isinstance(x, str) else x)
        .set_index("group")
    )
    return df


if __name__ == "__main__":
    main()
