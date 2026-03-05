"""Find untranscribed conversation files and bundle them."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGURATION - Set your parameters here
# ============================================================
AUDIO_FOLDER = "/mnt/Coral_NAS/raw/conversations"
TRANSCRIPTION_FOLDER = "/mnt/Coral_NAS/raw/transcriptions"
OUTPUT_FOLDER = "/mnt/Coral_NAS/raw/conversations_no_transcripts"
HOURS_PER_BUNDLE = 30

# Excel file configuration
CSV_FILE = "Filoversigt.csv"  # Path to your CSV file
FILENAME_COLUMN = "Filnavn"  # Column name containing filenames
STATUS_COLUMN = "Status"  # Column name containing status

# Statuses to exclude from processing
EXCLUDED_STATUSES = [
    "Venter På Tjek",
    "Fejl",
    "Pi",
    "I Gang",
]  # Files with these statuses will be skipped
# ============================================================


def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds using ffmpeg.

    Args:
        file_path (Path): Path to the audio file
    Returns:
        float: Duration of the audio file in seconds, or 0 if it cannot be determined.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(file_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data["format"]["duration"])
            return duration
    except Exception as e:
        print(f"Warning: Could not read duration for {file_path.name}: {e}")

    return 0


def load_excluded_files_from_csv():
    """Load filenames and their statuses from CSV file.

    Returns:
        set of str: Set of filenames (without extension) that should be excluded based on their status.
    """
    try:
        # Read CSV file
        df = pd.read_csv(CSV_FILE)

        excluded_files = set()

        print(f"\nLoading exclusion list from CSV: {CSV_FILE}")
        print(f"Excluded statuses: {', '.join(EXCLUDED_STATUSES)}")

        # Check if required columns exist
        if FILENAME_COLUMN not in df.columns:
            print(f"Warning: Column '{FILENAME_COLUMN}' not found in CSV file")
            return set()

        if STATUS_COLUMN not in df.columns:
            print(f"Warning: Column '{STATUS_COLUMN}' not found in CSV file")
            return set()

        # Process rows
        for _, row in df.iterrows():
            filename = str(row[FILENAME_COLUMN]).strip()
            status = str(row[STATUS_COLUMN]).strip()

            if status in EXCLUDED_STATUSES and filename and filename != "nan":
                # Remove extension if present
                filename_stem = Path(filename).stem
                excluded_files.add(filename_stem)

        print(f"Found {len(excluded_files)} files to exclude based on status")
        return excluded_files

    except FileNotFoundError:
        print(f"Warning: CSV file '{CSV_FILE}' not found")
        print("Continuing without exclusion list...")
        return set()
    except Exception as e:
        print(f"Warning: Could not load CSV file: {e}")
        print("Continuing without exclusion list...")
        return set()


def find_unmatched_audio(audio_folder, transcription_folder, excluded_files=None):
    """Find audio files that don't have corresponding transcription files.

    Args:
        audio_folder (str or Path): Path to folder containing audio files
        transcription_folder (str or Path): Path to folder containing .ass transcription files
        excluded_files (set of str, optional): Set of filenames (without extension) to exclude

    Returns:
        Tuple of (unmatched files list, excluded count, total unmatched before exclusion)
    """
    if excluded_files is None:
        excluded_files = set()

    # Common audio file extensions
    audio_extensions = {
        ".mp3",
        ".wav",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".wma",
        ".aiff",
        ".ape",
        ".alac",
    }

    audio_path = Path(audio_folder)
    audio_files = [
        f
        for f in audio_path.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]

    trans_path = Path(transcription_folder)
    trans_files = [
        f for f in trans_path.iterdir() if f.is_file() and f.suffix.lower() == ".ass"
    ]

    audio_dict = {f.stem: f for f in audio_files}
    trans_names = {f.stem for f in trans_files}

    unmatched = []
    excluded_count = 0
    total_unmatched = 0

    for stem, file_path in audio_dict.items():
        # Check if transcription is missing
        if stem not in trans_names:
            total_unmatched += 1

            # Skip if file is in exclusion list
            if stem in excluded_files:
                excluded_count += 1
                continue

            unmatched.append(file_path)

    return sorted(unmatched, key=lambda x: x.name), excluded_count, total_unmatched


def bundle_audio_files(audio_files, output_folder, hours_per_bundle=30):
    """Copy audio files to output folder, organizing them into bundles of specified duration.

    Args:
        audio_files (list of Path): List of Path objects for audio files
        output_folder (str or Path): Path to output folder
        hours_per_bundle (int): Target duration in hours for each bundle
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    seconds_per_bundle = hours_per_bundle * 3600

    current_bundle = 1
    current_duration = 0
    bundle_files: list[Path] = []

    print("\nProcessing files and calculating durations...")

    for audio_file in audio_files:
        duration = get_audio_duration(audio_file)

        # If adding this file would exceed the bundle limit, create a new bundle
        if current_duration > 0 and current_duration + duration > seconds_per_bundle:
            # Create the current bundle
            bundle_folder = output_path / f"bundle_{current_bundle:03d}"
            bundle_folder.mkdir(exist_ok=True)

            bundle_hours = current_duration / 3600
            print(
                f"\nBundle {current_bundle}: {len(bundle_files)} files, {bundle_hours:.2f} hours"
            )

            # Copy files with progress bar
            for src_file in tqdm(
                bundle_files, desc=f"Copying bundle {current_bundle}", unit="file"
            ):
                dest_file = bundle_folder / src_file.name
                shutil.copy2(src_file, dest_file)

            # Start new bundle
            current_bundle += 1
            current_duration = 0
            bundle_files = []

        # Add file to current bundle
        bundle_files.append(audio_file)
        current_duration += duration

    # Copy remaining files in the last bundle
    if bundle_files:
        bundle_folder = output_path / f"bundle_{current_bundle:03d}"
        bundle_folder.mkdir(exist_ok=True)

        bundle_hours = current_duration / 3600
        print(
            f"\nBundle {current_bundle}: {len(bundle_files)} files, {bundle_hours:.2f} hours"
        )

        # Copy files with progress bar
        for src_file in tqdm(
            bundle_files, desc=f"Copying bundle {current_bundle}", unit="file"
        ):
            dest_file = bundle_folder / src_file.name
            shutil.copy2(src_file, dest_file)

    print(f"\n{'='*60}")
    print(f"Created {current_bundle} bundle(s) in '{output_folder}'")


def main():
    """Main function to find unmatched audio files and bundle them."""
    print("Audio File Matcher and Bundler")
    print("=" * 60)

    # Use configured parameters
    audio_folder = AUDIO_FOLDER
    transcription_folder = TRANSCRIPTION_FOLDER
    output_folder = OUTPUT_FOLDER
    hours_per_bundle = HOURS_PER_BUNDLE

    if not os.path.isdir(audio_folder):
        print(f"Error: Audio folder '{audio_folder}' does not exist.")
        return

    if not os.path.isdir(transcription_folder):
        print(f"Error: Transcription folder '{transcription_folder}' does not exist.")
        return

    # Load excluded files from CSV
    excluded_files = load_excluded_files_from_csv()

    # Find unmatched files
    unmatched, excluded_count, total_unmatched = find_unmatched_audio(
        audio_folder, transcription_folder, excluded_files
    )

    print(f"\n{'='*60}")
    print(f"Total audio files without transcriptions: {total_unmatched}")
    print(f"Excluded files (based on CSV status): {excluded_count}")
    print(f"Files to process: {len(unmatched)}")
    print(f"{'='*60}\n")

    if not unmatched:
        print("All audio files have corresponding transcriptions!")
        return

    # Show first few files
    print("First 10 unmatched files:")
    for filename in unmatched[:10]:
        print(f"  - {filename.name}")
    if len(unmatched) > 10:
        print(f"  ... and {len(unmatched) - 10} more")

    print(f"\nOutput folder: {output_folder}")
    print(f"Hours per bundle: {hours_per_bundle}")

    # Prompt user for confirmation
    user_input = input("\nProceed with copying files? (yes/no): ").strip().lower()
    if user_input not in ["yes", "y"]:
        print("Operation cancelled by user.")
        return

    # Bundle and copy files
    bundle_audio_files(unmatched, output_folder, hours_per_bundle)

    print("\nDone!")


if __name__ == "__main__":
    main()
