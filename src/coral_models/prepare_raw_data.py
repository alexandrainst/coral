"""Functions for preparing the raw data"""

import datetime
import sqlite3
import subprocess
from pathlib import Path
from zlib import adler32

import pandas as pd
import pycountry
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from tqdm import tqdm

DB_TO_EXCEL_METADATA_NAMES = {
    "name": "name",
    "email": "mail",
    "age": "age",
    "gender": "gender",
    "dialect": "dialect",
    "accent": "native_language",
    "languages": "spoken_languages",
    "zipcode_school": "zip_primary_school",
    "zipcode_birth": "zip_grew_up",
    "birth_place": "birthplace",
    "education": "education",
    "occupation": "occupation",
}


def make_speaker_metadata(raw_path: Path, metadata_path: Path) -> pd.DataFrame:
    """Make a speaker metadata dataframe from the raw data.

    Args:
        raw_path (Path):
            The path to the raw data.
        metadata_path (Path):
            The path to the metadata file.

    Returns:
        pd.DataFrame:
            The speaker metadata."""
    speaker_metadata = pd.read_excel(metadata_path, index_col=0)

    # Replace all nan values with empty strings, because any nan values are
    # due to the fact that the speaker did not provide the information in the
    # metadata form. All metadata columns are strings, hence we can replace
    # nan values with empty strings.
    speaker_metadata = speaker_metadata.fillna("")

    # Get the columns that contain information about speakers
    speaker_a_columns = [col for col in speaker_metadata.columns if "subject_a" in col]
    speaker_b_columns = [col for col in speaker_metadata.columns if "subject_b" in col]
    recorder_columns = [col for col in speaker_metadata.columns if "recorder" in col]

    # We have speaker_a and speaker_b columns, but we want a single speaker column
    speaker_a = speaker_metadata[speaker_a_columns].rename(
        lambda x: x.replace("subject_a_", ""), axis=1
    )
    speaker_b = speaker_metadata[speaker_b_columns].rename(
        lambda x: x.replace("subject_b_", ""), axis=1
    )
    recorder = speaker_metadata[recorder_columns].rename(
        lambda x: x.replace("recorder_", ""), axis=1
    )

    speaker_list = [speaker_a, speaker_b, recorder]

    # Load speaker information from read aloud data
    read_aloud_paths = raw_path.glob("*_oplæst_*")
    for read_aloud_path in read_aloud_paths:
        read_aloud_data = get_data_from_db(read_aloud_path)

        read_aloud_data_speakers = read_aloud_data[DB_TO_EXCEL_METADATA_NAMES.keys()]
        read_aloud_data_speakers = read_aloud_data_speakers.rename(
            columns=DB_TO_EXCEL_METADATA_NAMES
        )
        speaker_list.append(read_aloud_data_speakers)

    # Concatenate all speaker information
    speakers = pd.concat(speaker_list, axis=0).drop_duplicates()

    # People often typed their name in all lower case or similar variations, which
    # yields a lot of duplicates. We therefore remove rows with duplicate emails.
    speakers = speakers.drop_duplicates(subset=["mail"]).reset_index(drop=True)

    # The `native_language` column contains both alpha-2 codes and full names, so we
    # need to convert the alpha-2 codes to full names. The same goes for the
    # birthplace. `spoken_languages` is a list of languages, so we need to convert that
    # as well
    speakers["native_language"] = speakers["native_language"].apply(correct_country)
    speakers["birthplace"] = speakers["birthplace"].apply(correct_country)
    speakers["spoken_languages"] = speakers["spoken_languages"].apply(
        correct_country_list
    )

    # Specification of gender is not consistent, so we need to correct that
    # by mapping "K" to "female", and "M" to "male".
    speakers["gender"] = speakers["gender"].apply(
        lambda x: dict(M="male", K="female", m="male", k="female").get(x, x)
    )

    # Create a speaker id column.
    speakers["speaker_id"] = speakers[["name", "mail"]].apply(
        lambda x: speaker_id(x[0], x[1]), axis=1
    )
    return speakers


def make_recording_metadata(
    speakers: pd.DataFrame, raw_path: Path, sentences: pd.DataFrame, metadata_path: Path
) -> pd.DataFrame:
    """Make a recording metadata file from the raw data

    Args:
        speakers (pd.DataFrame):
            The speakers dataframe
        raw_path (Path):
            The path to the raw data
        sentences (pd.DataFrame):
            The sentences dataframe

    Returns:
        pd.DataFrame: The recording metadata dataframe."""
    metadata_path = raw_path / "metadata.xlsx"

    # Make speaker-metadata file
    speaker_metadata = pd.read_excel(metadata_path, index_col=0)

    # Get the columns that contain information about the recording environment
    # but not about the speakers and recorders, except their email and name, which
    # we need to create their speaker ids.
    cols_needed_for_ids = [
        "subject_a_name",
        "subject_b_name",
        "recorder_name",
        "subject_a_mail",
        "subject_b_mail",
        "recorder_mail",
    ]
    recording_metadata = speaker_metadata[
        [
            col
            for col in speaker_metadata.columns
            if not any(
                [
                    substring in col
                    for substring in ["subject_a", "subject_b", "recorder"]
                ]
            )
        ]
        + cols_needed_for_ids
    ]

    # People often typed their name in all lower case or similar variations, which
    # yields a lot of duplicates. We therefore remove rows with duplicate emails.
    mail_to_name = speakers.set_index("mail")["name"].to_dict()
    recording_metadata["subject_a_name"] = recording_metadata["subject_a_mail"].map(
        mail_to_name
    )
    recording_metadata["subject_b_name"] = recording_metadata["subject_b_mail"].map(
        mail_to_name
    )

    # Make a speaker_id column for the recording metadata, with ids separated by a comma
    # Subject A is the first speaker id, subject B is the second speaker id
    recording_metadata["speaker_id"] = recording_metadata[
        ["subject_a_name", "subject_a_mail", "subject_b_name", "subject_b_mail"]
    ].apply(lambda x: (speaker_id(x[0], x[1]) + "," + speaker_id(x[2], x[3])), axis=1)

    # Make a recorder_id column for the recording metadata
    recording_metadata["recorder_id"] = recording_metadata[
        ["recorder_name", "recorder_mail"]
    ].apply(lambda x: speaker_id(x[0], x[1]), axis=1)

    recording_metadata = recording_metadata.drop(columns=cols_needed_for_ids)

    # Make a sentence_id column for the recording metadata, sentence_id is -1 if the
    # recorder does not contain a sentences from the sentence dataframe, i.e. if
    # recording is a conversation.
    # Sentences are fixed throughout the project, so we do not need to use a hash
    # function to create a unique id for each sentence.
    recording_metadata["sentence_id"] = -1

    # We need filenames for later, when we want to create recording ids.
    recording_metadata["filename"] = recording_metadata.index.astype(str)

    # Load speaker information from read aloud data
    recording_metadata_list = []
    read_aloud_paths = raw_path.glob("*_oplæst_*")
    for read_aloud_path in tqdm(list(read_aloud_paths)):
        read_aloud_data = get_data_from_db(read_aloud_path)

        # Format filenames
        read_aloud_data["filename"] = read_aloud_data["recorded_file"].apply(
            lambda x: f"{read_aloud_path.parts[-1]}/audio_files/{x}"
        )

        # Start and stop times are just submitted times for read aloud data
        read_aloud_data["start"] = read_aloud_data["submitted_time"]
        read_aloud_data["stop"] = read_aloud_data["submitted_time"]

        # rename columns
        read_aloud_data = read_aloud_data.rename(
            columns={
                "room_dimensions": "dimensions",
                "background_noise": "background_noise_level",
            }
        )

        # Make sentence_id columns from content
        # We changed the set of sentences during the project, so we need to check if
        # the sentence is in the sentence list. If it is not, we set the sentence_id to
        # the next available id.
        read_aloud_data["sentence_id"] = -1
        for row_i, row in read_aloud_data.iterrows():
            if row["transcription"] not in sentences["text"].values:
                # Append new sentence to sentences dataframe
                sentences = pd.concat(
                    [
                        sentences,
                        pd.DataFrame(
                            {
                                "text": [row["transcription"]],
                                "sentence_id": len(sentences),
                            }
                        ),
                    ],
                    ignore_index=True,
                )

            read_aloud_data.loc[row_i, "sentence_id"] = sentences[
                sentences["text"] == row["transcription"]
            ].index[0]

        # Make a recorder_id. This is not in the read aloud data, as no we have
        # no information about the recorders for the read aloud data.
        read_aloud_data["recorder_id"] = -1

        read_aloud_data["speaker_id"] = "placeholder"

        # Append to recording metadata list
        recording_metadata_list.append(read_aloud_data)

    # Concatenate all read aloud recording metadata
    all_read_recording_metadata = (
        pd.concat(recording_metadata_list, axis=0)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # People often typed their name in all lower case or similar variations, which
    # yields a lot of duplicates. We therefore remove rows with duplicate emails.
    all_read_recording_metadata["name"] = all_read_recording_metadata["email"].map(
        mail_to_name
    )

    # Convert speaker emails to speaker IDs
    all_read_recording_metadata["speaker_id"] = all_read_recording_metadata[
        ["name", "email"]
    ].apply(lambda x: speaker_id(x[0], x[1]), axis=1)

    # Drop columns that are not in the recording metadata
    all_read_recording_metadata = all_read_recording_metadata[
        recording_metadata.columns
    ]

    # Concatenate the read aloud recording metadata with the other recording metadata
    all_recording_metadata = (
        pd.concat([all_read_recording_metadata, recording_metadata], axis=0)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Make a recording id column
    tqdm.pandas()
    all_recording_metadata["recording_id"] = all_recording_metadata[
        "filename"
    ].progress_apply(lambda x: recording_id(x, raw_path))

    # Remove rows with no recording id. Sometimes recorders did not submit their
    # all their recordings.
    all_recording_metadata = all_recording_metadata[
        all_recording_metadata["recording_id"].notna()
    ].reset_index(drop=True)

    # Prepend the sentence id column with "s"
    sentences["sentence_id"] = "s" + sentences.index.astype(str)
    all_recording_metadata["sentence_id"] = "s" + all_recording_metadata[
        "sentence_id"
    ].astype(str)

    # Start and Stop columns are in the format "HHMM-DD-MM-YY" and
    # "DD/MM/YYYY, HH:MM:SS" and need to be converted to
    # "DD/MM/YYYY HH:MM:SS+02:00"
    all_recording_metadata["start"] = all_recording_metadata["start"].apply(
        correct_timestamp
    )
    all_recording_metadata["stop"] = all_recording_metadata["stop"].apply(
        correct_timestamp
    )

    # The dimensions column has values in the format "X,Y,Z" and "XxYxZ". These need to
    #  be converted to "X,Y,Z"
    all_recording_metadata["dimensions"] = all_recording_metadata["dimensions"].apply(
        lambda x: ",".join(x.split("x"))
    )

    # The noise column has both "none", "ingen", "trafik", "traffic", which needs to be
    # only "none" or "traffic".
    all_recording_metadata["noise"] = all_recording_metadata["noise"].apply(
        lambda x: dict(ingen="none", trafik="traffic").get(x, x)
    )

    return all_recording_metadata, sentences


def prepare_raw_data(
    input_path: Path | str = Path("data/raw"),
    output_path: Path | str = Path("data/processed"),
    metadata_path: Path | str = Path("data/raw/metadata.csv"),
    hidden_output_path: Path | str = Path("data/hidden"),
) -> None:
    """Prepare the raw data.

    Args:
        input_path (Path or str, optional):
            Path to the raw data. Defaults to "data/raw".
        output_path (Path or str, optional):
            Path to the processed data. Defaults to "data/processed".
        metadata_path (Path or str, optional):
            Path to the metadata. Defaults to "data/raw/metadata.csv".
        hidden_input_path (Path or str, optional):
            Path to save sensitive information. Defaults to "data/hidden".
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    hidden_output_path = Path(hidden_output_path)

    # Make speaker-metadata dataframe
    speakers = make_speaker_metadata(input_path, metadata_path)

    # Make sentence-metadata dataframe
    sentences_path = input_path / "processed_articles.csv"
    sentences = pd.read_csv(sentences_path)

    # Make sentence-index, and simplify the sentence dataframe
    sentences["sentence_id"] = sentences.index
    sentences = sentences[["sentence_id", "text"]]

    # Make recording-metadata dataframe
    recordings, sentences = make_recording_metadata(
        speakers, input_path, sentences, metadata_path
    )

    # Check if the output path exists, and if not, make it
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # Make a folder for the processed audio files
    processed_audio_path = output_path / "processed_audio"

    if not processed_audio_path.exists():
        processed_audio_path.mkdir()

    # Convert the audio files to .wav and place them in the output path, and rename
    # their filenames in the recording metadata. We also calculate the duration of the
    # audio.
    rows_to_remove: list[int] = []

    def change_codec_and_rename_files(row: pd.Series, row_i: int) -> None:
        filename = input_path / row["filename"]

        # Check if the file is empty, and if it is, remove it from the dataframe
        # and continue to the next file
        if filename.stat().st_size < 10000:  # Any file smaller than this is empty
            rows_to_remove.append(row_i)
            return

        # Get the new filename
        # New filename is in the format is for conversations:
        # "recording_id_speaker_id1_speaker_id2_recorder_speaker_id_conversation.wav"
        # and for read aloud data:
        # "recording_id_speaker_id_sentence_id.wav"
        if len(row["speaker_id"].split(",")) > 1:
            speaker_id = "_".join(row["speaker_id"].split(","))
            sentence_id = "conversation"
            speaker_id += f"_{row['recorder_id']}"
        else:
            speaker_id = row["speaker_id"]
            sentence_id = row["sentence_id"]

        new_filename = (
            processed_audio_path
            / f"{row['recording_id']}_{speaker_id}_{sentence_id}.wav"
        )

        # Update the filename in the recording metadata
        recordings.loc[row_i, "filename"] = new_filename

        # If the file is an .webm file, convert it to .wav
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # overwrite output file if it exists
                "-i",  # input file
                str(filename),  # input file name
                "-acodec",  # audio codec
                "pcm_s16le",  # 16-bit signed integer PCM audio codec
                "-ac",  # number of audio channels
                "1",  # mono
                "-f",  # force format
                "wav",  # output format
                new_filename,  # output file name
            ],
            stdout=subprocess.DEVNULL,
        )

    # Convert the audio files and rename them
    for row_i, row in recordings.iterrows():
        change_codec_and_rename_files(row, row_i)

    # Remove rows with empty files
    recordings = recordings.drop(rows_to_remove).reset_index(drop=True)

    # Write a README file
    readme = make_readme()
    with open(output_path / "README.md", "w") as f:
        f.write(readme)

    # Save the dataframes
    speakers.to_excel(hidden_output_path / "speakers.xlsx")
    sentences.to_excel(output_path / "sentences.xlsx")
    recordings.to_excel(output_path / "recordings.xlsx")

    # Make a dataframe with statistics about the data
    speakers["age"] = speakers["age"].astype(int)
    data_stats = pd.DataFrame(
        {
            "Number of speakers": len(speakers),
            "Number of sentences": len(sentences),
            "Number of recordings": len(recordings),
            "People over 50": len(speakers[speakers["age"] > 50]),
            "People below 50 and above 18": len(
                speakers[(speakers["age"] < 50) & (speakers["age"] > 18)]
            ),
            "People below 18": len(speakers[speakers["age"] < 18]),
            "Number of woman": len(speakers[speakers["gender"] == "female"]),
            "Number of men": len(speakers[speakers["gender"] == "male"]),
            "Unique dialects": len(speakers["dialect"].unique()),
        },
        index=[0],
    )
    data_stats.to_excel(output_path / "data_stats.xlsx")


def correct_country(country: str) -> str:
    """Converts a country name or alpha-2 language code to a full country name

    Args:
        country (str): A country name or alpha-2 code

    Returns:
        str: The full country name
    """
    if len(country) == 2:
        country_obj = pycountry.countries.get(alpha_2=country)
    else:
        country_obj = None

    if country_obj is not None:
        return country_obj.name
    else:
        return country.capitalize()


def correct_country_list(country_list: str) -> str:
    """Converts a str containing a list of countries to a list of full country names

    Args:
        country_list (str): A string containing a list of countries

    Returns:
        str: A string containing a list of full country names.
    """
    for delimiter in [",", ";", " "]:
        if len(country_list.split(delimiter)) > 1:
            return ";".join(
                [
                    correct_country(country.strip())
                    for country in country_list.split(delimiter)
                ]
            )
    return correct_country(country_list)


def correct_timestamp(timestamp: str) -> str:
    """Uniformises a timestamp.

    Args:
        timestamp (str): The timestamp to convert

    Returns:
        str: The converted timestamp
    """
    if ":" in timestamp:
        format = "%d/%m/%Y, %H:%M:%S"
    else:
        format = "%H%M-%d-%m-%y"
    try:
        return datetime.datetime.strptime(timestamp, format).strftime(
            "%Y-%m-%dT%H:%M:%S+02:00"
        )
    except ValueError:
        return timestamp


def get_data_from_db(db_folder: Path) -> pd.DataFrame:
    """Gets the data from the database

    Args:
        db_folder (Path): The path to the folder containing the database
            the database should be named "db.sqlite3"

    Returns:
        pd.DataFrame: The data from the database
    """
    connection = sqlite3.connect(db_folder / "db.sqlite3")
    read_aloud_data = pd.read_sql_query(
        sql="SELECT * FROM CoRal_recording", con=connection
    )
    return read_aloud_data


def speaker_id(name: str, email: str) -> str:
    """Creates a speaker id from a name and email.

    We use the adler32 hash function on the speakers name and email to create a
    unique id. We use the adler32 hash function because it is fast and has a
    low collision rate for short strings, and produces hash-values which are
    integers with 8 or 9 digits, we use the first 8 digits as the id.
    Args:
        name (str): The name of the speaker
        email (str): The email of the speaker

    Returns:
        str: The speaker id
    """
    return "t" + str(adler32(bytes(name + email, "utf-8")))[0:8]


def recording_id(filename: str, data_folder: Path) -> str | None:
    """Creates a recording id from the content of the recording.

    We use the adler32 hash function on the raw data to create a unique id. We
    use the adler32 hash function because it is fast and has a low collision
    rate for short strings, and produces hash-values which are integers with 8
    or 9 digits, we use the first 8 digits as the id.
    If the recording cannot be decoded, we use the adler32 hash function on the
    filename instead.
    Args:
        filename (str): The filename of the recording

    Returns:
        str: The recording id
    """
    file_path = data_folder / filename
    try:
        return "r" + str(adler32(AudioSegment.from_file(file_path).raw_data))[0:8]
    except CouldntDecodeError:
        return "r" + str(adler32(bytes(filename, "utf-8")))[0:8]
    except FileNotFoundError:
        return None


def make_readme() -> str:
    """Makes a README.md file"""
    return """# CoRal data

    The CoRal data is a collection of recordings of people reading aloud and having
    conversations. The data was collected by the Alexandra Institute in 2023-2025.

    ## Data

    The recordings are stored in the `processed_audio` folder. The recordings are
    stored in the following format:

    `[recording_id]_[speaker_id1]_[speaker_id2]_[recorder_speaker_id]_conversation.wav`

    for conversations, and

    `[recording_id]_[speaker_id]_[sentence_id].wav`

    for read aloud data.

    A `recording_id` is a unique id for each recording. A `speaker_id` is a unique id
    for each speaker. The `recorder_speaker_id` is the unique `speaker_id` associated
    with each recorder, i.e. the person which performed the recording. A `sentence_id`
    is a unique id for each sentence. The prefix `conversation` indicates that a
    recording is of two people having a conversation.

    The metadata for the recordings are stored in the `recordings.csv` file. The
    metadata for the speakers are stored in the `speakers.csv` file. The metadata for
    the sentences are stored in the `sentences.csv` file.
    """
