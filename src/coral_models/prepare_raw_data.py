"""Functions for preparing the raw data"""

import contextlib
import datetime
import sqlite3
import subprocess
import wave
from pathlib import Path

import pandas as pd
import pycountry
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
        cfg (DictConfig):
            The Hydra configuration object.
        raw_path (Path):
            The path to the raw data."""
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
    speakers = pd.concat(speaker_list, axis=0).drop_duplicates().reset_index(drop=True)

    # People often typed their name in all lower case or similar variations, which
    # yields a lot of duplicates. We therefore remove rows with duplicate emails.
    speakers = speakers.drop_duplicates(subset=["mail"])

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
        lambda x: dict(M="male", K="female").get(x, x)
    )

    # Create a speaker id column
    speakers["speaker_id"] = "t" + speakers.index.astype(str)

    return speakers


def make_recording_metadata(
    speakers: pd.DataFrame, raw_path: Path, sentences: pd.DataFrame, metadata_path: Path
) -> pd.DataFrame:
    """Make a recording metadata file from the raw data

    Args:
        cfg (DictConfig):
            The config
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
    # but not about the speakers and recorders, except their email.
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
        + ["subject_a_mail", "subject_b_mail", "recorder_mail"]
    ]

    # Make a dict from speaker mail to speaker id in the speakers dataframe
    speaker_mail_to_id = dict(zip(speakers["mail"], speakers["speaker_id"]))

    # Make a speaker_id column for the recording metadata, with ids separated by a comma
    # Subject A is the first speaker id, subject B is the second speaker id
    recording_metadata["speaker_id"] = recording_metadata[
        ["subject_a_mail", "subject_b_mail"]
    ].apply(lambda x: ",".join([str(speaker_mail_to_id[mail]) for mail in x]), axis=1)

    # Make a recorder_id column for the recording metadata
    recording_metadata["recorder_id"] = recording_metadata["recorder_mail"].apply(
        lambda x: str(speaker_mail_to_id[x])
    )

    recording_metadata = recording_metadata.drop(
        columns=["subject_a_mail", "subject_b_mail", "recorder_mail"]
    )

    # Make a sentence_id column for the recording metadata, sentence_id is -1 if the
    # recorder does not contain a sentences from the sentence dataframe, i.e. if
    # recording is a conversation.
    recording_metadata["sentence_id"] = -1

    # Make a sentence content to sentence_id dict
    sentence_content_to_id = dict(zip(sentences["text"], sentences["sentence_id"]))

    # Load speaker information from read aloud data
    recording_metadata_list = [recording_metadata]
    read_aloud_paths = raw_path.glob("*_oplæst_*")
    for read_aloud_path in read_aloud_paths:
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

        # Make speaker mails in to speaker ids
        read_aloud_data["speaker_id"] = read_aloud_data["email"].map(speaker_mail_to_id)

        # Make sentence_id columns from content
        # We changed the set of sentences during the project, so we need to check if
        # the sentence is in the sentence list. If it is not, we set the sentence_id to
        # the next available id.
        read_aloud_data["sentence_id"] = -1
        for row_i, row in read_aloud_data.iterrows():
            if row["transcription"] in sentence_content_to_id.keys():
                read_aloud_data.loc[row_i, "sentence_id"] = sentence_content_to_id[
                    row["transcription"]
                ]
            else:
                sentence_content_to_id[row["transcription"]] = len(
                    sentence_content_to_id
                )
                read_aloud_data.loc[row_i, "sentence_id"] = sentence_content_to_id[
                    row["transcription"]
                ]

        # Make a recorder_id. This is not in the read aloud data, as no we have
        # no information about the recorders for the read aloud data.
        read_aloud_data["recorder_id"] = -1

        # Drop columns that are not in the recording metadata
        read_aloud_data = read_aloud_data[recording_metadata.columns]

        # Append to recording metadata list
        recording_metadata_list.append(read_aloud_data)

    # Concatenate all recording metadata
    all_recording_metadata = (
        pd.concat(recording_metadata_list, axis=0)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Make a recording id column
    all_recording_metadata["recording_id"] = "r" + all_recording_metadata.index.astype(
        str
    )

    # We have updated the sentence_content_to_id dict, so we also need to update the
    # sentences dataframe with the new sentence ids
    sentences = (
        pd.DataFrame.from_dict(sentence_content_to_id, orient="index")
        .reset_index()
        .rename(columns={"index": "text", 0: "sentence_id"})
    )

    # Prepend the sentence id column with "s"
    sentences["sentence_id"] = "s" + sentences.index.astype(str)
    all_recording_metadata["sentence_id"] = "s" + all_recording_metadata[
        "sentence_id"
    ].astype(str)

    # Start and Stop columns are in the format "HHMM-DD-MM-YY" and
    # "DD/MM/YYYY HH:MM:SS" and need to be converted to
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
    all_recording_metadata["background_noise_level"] = all_recording_metadata[
        "background_noise_level"
    ].apply(lambda x: dict(ingen="none", trafik="traffic").get(x, x))

    return all_recording_metadata, sentences


def prepare_raw_data(
    input_path_str: str = "data/raw",
    output_path_str: str = "data/processed",
    metadata_path_str: str = "data/raw/metadata.csv",
):
    """Prepare the raw data.

    Args:
        input_path (Path, optional): Path to the raw data. Defaults to "data/raw".
        output_path (Path, optional): Path to the processed data.
            Defaults to "data/processed".
        metadata_path (Path, optional): Path to the metadata. Defaults to
            "data/raw/metadata.csv".
    """
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    metadata_path = Path(metadata_path_str)

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
    read_aloud_duration = 0.0
    conversation_duration = 0.0
    for row_i, row in tqdm(recordings.iterrows()):
        # Get the filename
        filename = input_path / row["filename"]

        # Get the new filename
        # New filename is in the format is for conversations:
        # "recording_id_speaker_id1_speaker_id2_recorder_speaker_id_conversation.wav"
        # and for read aloud data:
        # "recording_id_speaker_id_recorder_id_sentence_id.wav"
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

        # Get the duration of each audio file
        try:
            new_filename_str = str(new_filename)
            with contextlib.closing(wave.open(str(new_filename_str), "r")) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                if "conversation" in new_filename_str:
                    conversation_duration += duration
                else:
                    read_aloud_duration += duration
        except FileNotFoundError:
            pass

    # Make a dataframe with statistics about the data
    data_stats = pd.DataFrame(
        {
            "Number of speakers": len(speakers),
            "Number of sentences": len(sentences),
            "Number of recordings": len(recordings),
            "Read aloud duration (hours)": round(read_aloud_duration / 3600, 2),
            "Conversation duration (hours)": round(conversation_duration / 3600, 2),
            "Total duration (hours)": round(
                (read_aloud_duration + conversation_duration) / 3600, 2
            ),
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

    # Save the dataframes
    data_stats.to_csv(output_path / "data_stats.csv", index=False)
    speakers.to_csv(output_path / "speakers.csv", index=False)
    sentences.to_csv(output_path / "sentences.csv", index=False)
    recordings.to_csv(output_path / "recordings.csv", index=False)


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
    """Converts a timestamp from "HHMM-DD-MM-YY" to "DD/MM/YYYY HH:MM:SS+02:00"

    Args:
        timestamp (str): The timestamp to convert

    Returns:
        str: The converted timestamp
    """
    if ":" in timestamp:
        format = "DD/MM/YYYY HH:MM:SS"
    else:
        format = "%H%M-%d-%m-%y"
    try:
        return datetime.datetime.strptime(timestamp, format).strftime(
            "YYYY-MM-DDTHH:MM:SS+02:00"
        )
    except ValueError:
        return timestamp


def get_data_from_db(db_folder: Path) -> pd.DataFrame:
    connection = sqlite3.connect(db_folder / "db.sqlite3")
    read_aloud_data = pd.read_sql_query(
        sql="SELECT * FROM CoRal_recording", con=connection
    )
    return read_aloud_data
