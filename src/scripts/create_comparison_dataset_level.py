"""Creates a plot comparing the performance of different models on different datasets.

Usage:
    python src/scripts/create_comparison_dataset_level.py \
        -d DATASET_NAME \
        -m MODEL_NAME \
        [-d DATASET_NAME ...] \
        [-m MODEL_NAME ...] \
        [--metric METRIC]

DATASET_NAME: The dataset name needs to be the name of the folder stored in outputs/results/
             when running evaluation scripts with the dataset. 
MODEL_NAME: Should be the name of the base name of the model without paths and extra. The code 
            searches for the name in the score files inside the dataset results folder.
You can add as many models and datasets as needed.

OBS! Duplicates of score files with same model name will create problems.
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.style.use("ggplot")


METRIC_NAMES = dict(cer="Character Error Rate", wer="Word Error Rate")

# Conversion dict for different dataset names to use in the plots
DATASET_NAMES = {
    "alexandrainst-coral":"CoRal-v1",
    "alexandrainst-nst-da":"NST-da", 
    "Alvenir-alvenir_asr_da_eval-oss":"AlvenirOss",
    "Alvenir-alvenir_asr_da_eval-wiki":"AlvenirWiki",
    "google-fleurs":"Fleurs-da_dk",
    "mozilla-foundation-common_voice_17_0":"CommonVoice17",
    "Alvenir-conversational-finance": "AlvenirConvFinance",
    "CoRal-dataset-coral-v2-conv": "CoRal-v2-conv",
    "CoRal-dataset-coral-v2-read": "CoRal-v2-read"
}

@click.command()
@click.option(
    "--model",
    "-m",
    multiple=True,
    help="The name of a model that has been evaluated with the `evaluate_model` script.",
    required=True,
)
@click.option(
    "--dataset",
    "-d",
    multiple=True,
    help="The name of a dataset the models have been evaluated on with the `evaluate_model` script.",
    required=True,
)
@click.option(
    "--metric",
    default="cer",
    type=click.Choice(["cer", "wer"]),
    help="The metric to plot.",
)
@click.option(
    "--dir",
    default=Path("outputs/results"),
    type=click.Path(exists=True),
    help="The path to the directory with results structured into folders with eval datasets",
)
def main(dataset: tuple[str], model: tuple[str], metric: str, dir: Path) -> None:
    """Creates a plot comparing the performance of different models on a dataset.

    Args:
        dataset:
            A tuple of dataset names to include in plot
        model:
            A tuple of models to include in plot
        metric:
            The metric to plot. Either "cer" or "wer".
    """
    os.makedirs("outputs/vis/comparisons", exist_ok=True)

    datasets = list(dataset)
    models = list(model)

    dfs = dict()

    for dataset in datasets:
        dataset_path = dir.joinpath(dataset)
        if dataset_path.is_dir():
            p = dataset_path.glob('**/*scores.csv')
            csvs = [x for x in p if x.is_file()]

            for model in models:
                model_csvs = [x for x in csvs if model in str(x)]

                if len(model_csvs)>1:
                    # check if one option is just the stem of the model without any subsets
                    #if exists then use only this model in model_csvs
                    # This is important if model names are extensions of each other
                    len_model_csvs = [len(str(model_csv).split("-")) for model_csv in model_csvs]
                    min_len = min(len_model_csvs)
                    model_csvs = [model_csv for model_csv, length in zip(model_csvs, len_model_csvs) if length == min_len]

                model=model.replace("oe", "ø")

                if not model in dfs.keys():
                    dfs[model] = dict()
                    
                if len(model_csvs)==1:
                    model_path = model_csvs[0]
                    dfs[model][dataset] = load_overall_score(file=model_path)[metric]
                elif len(model_csvs)>1:
                    for model_csv in model_csvs:
                        dataset_subset = model_csv.stem.split('-')[-2]
                        model_path = model_csv

                        dfs[model][dataset+'-'+dataset_subset] = load_overall_score(file=model_path)[metric]
                else:
                    print(f"Model csv for {model} and dataset {dataset} does not exist")
        else:
            print(f"Dataset {dataset_path} does not exist")

    df = pd.DataFrame.from_records(
        dfs
    )
    df.index = df.index.map(DATASET_NAMES)

    sorted_columns = sort_models(df.columns)
    df = df[sorted_columns]
    
    title=f"{METRIC_NAMES[metric.lower()]} by Group (Lower is Better)"
    df.plot(
        kind="bar",
        title=title,
        ylabel=METRIC_NAMES[metric.lower()],
        legend=True,
        figsize=(12, 6),
        rot=25,
    )
    plt.tight_layout(pad=2)
    plt.savefig(f"outputs/vis/comparisons/{metric}_comparison.png")
    plt.show()

    plt.savefig("results_vis.png")

    convert_to_markdown(df, title)

def load_overall_score(file: Path) -> pd.DataFrame:
    """Loads the evaluation data from a CSV file into a pandas DataFrame.

    Args:
        file:
            The path to the evaluation file.

    Returns:
        A pandas DataFrame containing the overall score of the evaluation data.
    """
    raw_df = pd.read_csv(file, index_col=None)
    return raw_df.iloc[-1,:]

def convert_to_markdown(df, title):
    """Convert DataFrame to markdown table format."""
    print(f"## {title} Results")
    print("|  | " + " | ".join(df.columns[1:]) + " |")
    print("|" + ":---:|" * len(df.columns))
    for idx, row in df.iterrows():
        print("|", idx, "|", " | ".join(f"{float(val)*100:.1f}" for val in row[1:]), "|")

def load_coral_evaluation_df(file: Path) -> pd.DataFrame:
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

def is_unstructured_model(model_name):
    # Check if the model name doesn't match the expected structured pattern
    parts = model_name.split('-')
    return len(parts) < 4 or parts[1] not in ['whisper', 'wav2vec2']

def parse_model_name(model_name):
    if is_unstructured_model(model_name):
        # Return a special tuple that sorts these models to the front
        return (0, 0, 0) 

    parts = model_name.split('-')
    model_type = parts[1]
    size = parts[2]
    version = parts[-1]
    
    if model_type == 'whisper':
        size_mapping = {
            'small': 1,
            'medium': 2,
            'large': 3
        }
    else:
        size_mapping = {
            '315m': 1,
            '1b': 2,
            '2b': 3
        }
    
    version_number = int(version[1:])

    # Returns tuple that facilitates sorting
    return (1 if model_type == 'whisper' else 2, size_mapping[size.lower()], version_number)

def sort_models(models):
    def custom_key(model_name):
        return parse_model_name(model_name)
    
    return sorted(models, key=custom_key)

if __name__ == "__main__":
    main()
