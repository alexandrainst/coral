"""Creates a histogram of the levenshtein distance between predictions and targets of erroneous
predictions of a model on different datasets.  

Usage:
    python src/scripts/create_levenshtein_plots.py \
        -d DATASET_NAME \
        -m MODEL_NAME \
        [-d DATASET_NAME ...] \
        [-m MODEL_NAME ...] \
        [--metric METRIC]

DATASET_NAME: The dataset name needs to be the name of the folder stored in outputs/results/
             when running evaluation scripts with the dataset. 
MODEL_NAME: Should be the name of the base name of the model without paths and extra. The code 
            searches for the name in the prediction files inside the dataset results folder.
You can add as many models and datasets as needed. There will be created one plot per model
including all the datasets in each plot.

OBS! Multiple prediction files with same model name will create problems.

"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import Levenshtein
import numpy as np

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
    "--dir",
    default=Path("outputs/results"),
    type=click.Path(exists=True),
    help="The path to the directory with results structured into folders with eval datasets",
)
def main(dataset: tuple[str], model: tuple[str], dir: Path) -> None:
    plot_stacked_histograms(dataset, model, dir)

def plot_stacked_histograms(dataset: tuple[str], model: tuple[str], dir: Path):
    """Creates a plot comparing the performance of different models on a dataset.

    Args:
        dataset:
            A tuple of dataset names to include in plot
        model:
            A tuple of models to include in plot
        dir:
            Path within dataset folders with prediction csvs can be found
    """
    datasets = list(dataset)
    models = list(model)

    dfs = dict()

    for dataset in datasets:
        dataset_path = dir.joinpath(dataset)
        if dataset_path.is_dir():
            p = dataset_path.glob('**/*predictions.csv')
            csvs = [x for x in p if x.is_file()]

            for model in models:
                model_csvs = [x for x in csvs if model in str(x)]
                model=model.replace("oe", "ø")

                if not model in dfs.keys():
                    dfs[model] = dict()
                    
                if len(model_csvs)==1:
                    model_path = model_csvs[0]
                    leven = load_preds_and_targets_calculate_levenshtein(model_path)
                    dfs[model][dataset] = leven[leven>0]
                elif len(model_csvs)>1:
                    for model_csv in model_csvs:
                        dataset_subset = model_csv.stem.split('-')[-2]
                        model_path = model_csv
                        leven = load_preds_and_targets_calculate_levenshtein(model_path)
                        dfs[model][dataset+'-'+dataset_subset] = leven[leven>0]
                else:
                    print(f"Prediction csv for {model} and dataset {dataset} does not exist")
        else:
            print(f"Dataset {dataset_path} does not exist")

    df = pd.DataFrame.from_records(
        dfs
    )
    df.index = df.index.map(DATASET_NAMES)

    sorted_columns = sort_models(df.columns)
    df = df[sorted_columns]
    
    for model in df.columns:
        print(f"Creating levenshtein plot for {model}")
        datasets_series = {dataset: df.loc[dataset, model] for dataset in df.index}
        
        bins = 20
        all_series= [val for val in datasets_series.values() if isinstance(val, pd.Series)] #Ignore datasets missing prediction file
        all_mistakes = [] 
        if all_series:
            all_data = pd.concat(all_series)

            std_multiplier = 3
            mean = all_data.mean()
            std_dev = all_data.std()
            threshold = mean + std_multiplier * std_dev

            # Filter out data above the threshold
            datasets_series_filtered = {
                dataset: data_series[data_series <= threshold] 
                for dataset, data_series in datasets_series.items() if isinstance(data_series, pd.Series)
            }         
            # Calculate a new data range based on filtered data
            combined_filtered_data = pd.concat(datasets_series_filtered.values())
            data_range_filtered = (combined_filtered_data.min(), combined_filtered_data.max())
            
            cumulative_data = []

            for data_series in datasets_series.values():
                hist, bin_edges = np.histogram(data_series, bins=bins, range=data_range_filtered)
                cumulative_data.append(hist)

            plt.figure(figsize=(10, 6))
            plt.hist(cumulative_data, bins=bin_edges, label=datasets_series.keys(), stacked=True)
            plt.title(f'Stacked Histogram for {model}')
            plt.xlabel('Levenshtein distance')
            plt.ylabel('Frequency')
            plt.legend(title='Datasets')
            plt.tight_layout(pad=2)
            plt.savefig(f"outputs/vis/levenshtein/levenshtein_{model}.png")
            plt.close()


def load_preds_and_targets_calculate_levenshtein(file: Path) -> pd.DataFrame:
    """Loads the prediction data from a CSV file into a pandas DataFrame.

    Args:
        file:
            The path to the prediction file.

    Returns:
        A pandas DataFrame containing the evaluation data.
    """
    raw_df = pd.read_csv(file, index_col=None)
    return raw_df.apply(lambda x: Levenshtein.distance(str(x.predictions),str(x.labels)), axis=1)


def convert_to_markdown(df, title):
    """Convert DataFrame to markdown table format."""
    print(f"## {title} Results")
    print("|  | " + " | ".join(df.columns[1:]) + " |")
    print("|" + ":---:|" * len(df.columns))
    for idx, row in df.iterrows():
        print("|", idx, "|", " | ".join(f"{float(val)*100:.1f}" for val in row[1:]), "|")


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
