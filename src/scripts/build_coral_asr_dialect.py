"""Filter and prepare the CoRal ASR dataset for a specific Danish dialect."""

import logging
import os
import warnings

import hydra
import pandas as pd
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, load_dataset
from huggingface_hub import HfFolder, create_repo, upload_folder
from omegaconf import DictConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("roest-asr-demo")

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------ CONFIGURATION ------------ #
DIALECT_FILTER = "Nordjysk"  # e.g. "Sønderjysk", "Bornholmsk"
DATASET_NAME = "CoRal-project/coral-v2"
SUBSETS = ["read_aloud", "conversational"]
NEW_REPO_ID = f"alexandrainst/coral-v2-{DIALECT_FILTER.lower().replace(' ', '_')}"
OUTPUT_BASE = f"coral_v2_filtered_{DIALECT_FILTER.lower().replace(' ', '_')}"
HF_TOKEN = HfFolder.get_token()
# --------------------------------------- #


def filter_and_save_subset(subset):
    """Load a specific subset of the CoRal dataset, filter by dialect.

    Args:
        subset (str): The subset to filter (e.g., "read_aloud", "conversational").
    """
    print(f"\n📦 Loading full subset: {subset}")
    dsd = load_dataset(DATASET_NAME, subset)  # DatasetDict with train/val/test

    filtered_splits = {}
    for split_name, ds in dsd.items():
        print(f"🔍 Filtering '{split_name}' split...")
        filtered = ds.filter(lambda x: x["dialect"] == DIALECT_FILTER)
        print(f"✅ {len(filtered)} samples after filtering dialect='{DIALECT_FILTER}'")

        if len(filtered) == 0:
            continue

        records = []
        output_split_dir = os.path.join(OUTPUT_BASE, subset, split_name)
        audio_dir = os.path.join(output_split_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        id_column = "id_recording" if "read_aloud" in subset else "id_conversation"

        for ex in filtered:
            rec_id = ex[id_column]
            audio_filename = f"{rec_id}.flac"
            audio_rel_path = os.path.join("audio", audio_filename)
            audio_full_path = os.path.join(audio_dir, audio_filename)

            sf.write(
                audio_full_path, ex["audio"]["array"], ex["audio"]["sampling_rate"]
            )

            records.append(
                {
                    "id_recording": rec_id,
                    "dialect": ex["dialect"],
                    "text": ex["text"],
                    "audio": audio_rel_path,
                }
            )

        # Save metadata
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(output_split_dir, "metadata.csv"), index=False)

        # Convert to Dataset object
        ds_filtered = Dataset.from_pandas(df)
        ds_filtered = ds_filtered.cast_column("audio", Audio())
        filtered_splits[split_name] = ds_filtered

    if filtered_splits:
        # Save full subset
        subset_path = os.path.join(OUTPUT_BASE, subset)
        ds_dict = DatasetDict(filtered_splits)
        ds_dict.save_to_disk(subset_path)
    else:
        print(f"⚠️ No data for dialect='{DIALECT_FILTER}' in subset='{subset}'.")


def push_to_huggingface():
    """Upload the filtered dataset to Hugging Face Hub."""
    print(f"\n🚀 Uploading dataset to Hugging Face Hub: {NEW_REPO_ID}")
    create_repo(
        NEW_REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=True,  # Set to False if you want it public
    )

    upload_folder(
        folder_path=OUTPUT_BASE,
        repo_id=NEW_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print("✅ Upload complete!")


@hydra.main(
    config_path="../../config",
    config_name="dataset_dialect_creation",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Main function to filter and upload the CoRal dataset."""
    for subset in SUBSETS:
        filter_and_save_subset(subset)
    push_to_huggingface()


if __name__ == "__main__":
    main()
