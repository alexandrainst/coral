---
language:
- da
license: openrail
size_categories:
- 1K<n<10K
task_categories:
- automatic-speech-recognition
- audio-classification
pretty_name: CoRal Full Conversations (Redacted)
dataset_info:
- config_name: v3_full_conversation_redacted
  features:
  - name: id_conversation
    dtype: string
  - name: location
    dtype: string
  - name: location_roomdim
    dtype: string
  - name: noise_level
    dtype: string
  - name: noise_type
    dtype: string
  - name: audio
    dtype: audio
  - name: transcription_segments
    sequence:
    - name: speaker_id
      dtype: string
    - name: start_seconds
      dtype: float64
    - name: end_seconds
      dtype: float64
    - name: text
      dtype: string
    - name: is_redacted
      dtype: bool
  - name: speaker_a_id
    dtype: string
  - name: speaker_a_age
    dtype: string
  - name: speaker_a_gender
    dtype: string
  - name: speaker_a_dialect
    dtype: string
  - name: speaker_a_country_birth
    dtype: string
  - name: speaker_a_education
    dtype: string
  - name: speaker_a_occupation
    dtype: string
  - name: speaker_b_id
    dtype: string
  - name: speaker_b_age
    dtype: string
  - name: speaker_b_gender
    dtype: string
  - name: speaker_b_dialect
    dtype: string
  - name: speaker_b_country_birth
    dtype: string
  - name: speaker_b_education
    dtype: string
  - name: speaker_b_occupation
    dtype: string
  - name: recorder_id
    dtype: string
  - name: recorder_age
    dtype: string
  - name: recorder_gender
    dtype: string
  - name: recorder_dialect
    dtype: string
  - name: recorder_country_birth
    dtype: string
  - name: recorder_education
    dtype: string
  - name: recorder_occupation
    dtype: string
  splits:
  - name: train
    num_examples: -1
  download_size: -1
  dataset_size: -1
configs:
- config_name: v3_full_conversation_redacted
  data_files:
  - split: train
    path: v3_full_conversation_redacted/train-*
---

![alt text](https://alexandra.dk/wp-content/uploads/2025/04/CoRal_logo-01-768x200.png)

# CoRal: Full Conversations (Redacted)

Version 3.0


## Dataset Overview

This dataset contains the **full, unsplit conversations** from the [CoRal](https://huggingface.co/datasets/CoRal-project/coral-v3) Danish speech corpus. Unlike the main CoRal dataset, where conversations are split into individual utterances, this release preserves each conversation as a single audio file (typically 5-30 minutes long).

All personal information (PI) has been redacted: audio segments containing PI are replaced with silence, and the corresponding transcription segments are marked accordingly.


### Key Features

- **Full-length conversations**: Each sample is a complete conversation between two speakers, preserving natural turn-taking and conversational flow.
- **Privacy-preserving**: Personal information is silenced in the audio and flagged in the transcription segments.
- **Structured transcriptions**: Transcriptions are stored as time-aligned segments with speaker IDs, timestamps, and redaction flags — no external subtitle files needed.
- **Rich metadata**: Speaker demographics (age, gender, dialect, country of birth, education, occupation) and recording conditions (location, room dimensions, noise) are included per conversation.


### Quick Start

Due to the size of the audio files, streaming mode is recommended:

```python
from datasets import load_dataset

ds = load_dataset(
    "CoRal-project/coral_full_conversations",
    name="v3_full_conversation_redacted",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

sample = next(iter(ds))
print(f"Conversation: {sample['id_conversation']}")
print(f"Duration: {len(sample['audio']['array']) / sample['audio']['sampling_rate']:.0f} seconds")
print(f"Segments: {len(sample['transcription_segments']['text'])}")
```


## Data Fields

### Conversation-level

- `id_conversation`: Unique identifier for the conversation.
- `location`: Address of the recording location.
- `location_roomdim`: Dimensions of the recording room.
- `noise_level`: Noise level in the room (dB).
- `noise_type`: Type of noise the speakers were exposed to during recording. Note that the noise is not present in the audio.
- `audio`: The full conversation audio file (WAV format), with personal information replaced by silence.

### Transcription segments

Each conversation contains a list of time-aligned transcription segments (`transcription_segments`):

- `speaker_id`: Anonymized speaker identifier.
- `start_seconds`: Start time of the segment in seconds.
- `end_seconds`: End time of the segment in seconds.
- `text`: Transcribed text of the segment.
- `is_redacted`: Whether this segment contained personal information (if `true`, the corresponding audio has been silenced).

### Speaker metadata

Metadata is provided for all three participants — Speaker A, Speaker B, and the Recorder — with the following fields per role (prefixed with `speaker_a_`, `speaker_b_`, or `recorder_`):

- `*_id`: Anonymized speaker identifier.
- `*_age`: Age of the speaker.
- `*_gender`: Gender of the speaker.
- `*_dialect`: Self-reported dialect of the speaker.
- `*_country_birth`: Country where the speaker was born.
- `*_education`: Education level of the speaker.
- `*_occupation`: Occupation of the speaker.


## Data Statistics

We refer to [the full CoRal dataset](https://huggingface.co/datasets/CoRal-project/coral-v3) for statistics

## Example Datapoint

Below is a condensed example of a single datapoint (most segments omitted for brevity):

```json
{
  "id_conversation": "conv_07f9708fc0b8316a9dea85d473db112b",
  "location": "Krystalgade 15 1172 København",
  "location_roomdim": "325,280,270",
  "noise_level": "42",
  "noise_type": "human",
  "audio": { "array": [0.0, 0.001, -0.002, ...], "sampling_rate": 48000 },
  "transcription_segments": [
    {
      "speaker_id": "spe_755644b4408ff7b36e962ffa3ece6dd6",
      "start_seconds": 2.48,
      "end_seconds": 4.51,
      "text": "Jeg subjekt A og jeg hedder Veronica",
      "is_redacted": false
    },
    {
      "speaker_id": "spe_f7b23539162f9294ab83c6e5ca93a298",
      "start_seconds": 4.83,
      "end_seconds": 6.6,
      "text": "***PI***",
      "is_redacted": true
    },
    {
      "speaker_id": "spe_755644b4408ff7b36e962ffa3ece6dd6",
      "start_seconds": 10.12,
      "end_seconds": 13.31,
      "text": "Skal jeg tage den okay øh",
      "is_redacted": false
    }
  ],
  "speaker_a_id": "spe_755644b4408ff7b36e962ffa3ece6dd6",
  "speaker_a_age": "26",
  "speaker_a_gender": "female",
  "speaker_a_dialect": "nordsjællandsk",
  "speaker_a_country_birth": "DK",
  "speaker_a_education": "Gymnasiel Uddannelse",
  "speaker_a_occupation": "Studerende",
  "speaker_b_id": "spe_f7b23539162f9294ab83c6e5ca93a298",
  "speaker_b_age": "56",
  "speaker_b_gender": "female",
  "speaker_b_dialect": "amagermål",
  "speaker_b_country_birth": "IN",
  "speaker_b_education": "mellemlang_videregående_uddannelse",
  "speaker_b_occupation": "Børnehavepædagog",
  "recorder_id": "spe_e2151525032660a0b25b929739dca93d",
  "recorder_age": "29",
  "recorder_gender": "male",
  "recorder_dialect": "amagermål",
  "recorder_country_birth": "DK",
  "recorder_education": "lang_videregående_uddannelse",
  "recorder_occupation": "Konnsulent"
}
```

The second segment shows a redacted entry — `is_redacted` is `true`, the `text` is replaced with `***PI***`, and the corresponding audio interval (4.83s–6.6s) has been silenced.


## Relationship to CoRal

This dataset is derived from the same recordings as the `conversation` config in [CoRal v3](https://huggingface.co/datasets/CoRal-project/coral-v3). The key differences are:

| | CoRal `conversation` | This dataset |
|---|---|---|
| **Granularity** | Individual utterances | Full conversations |
| **Audio** | Short clips per utterance | Full recording per conversation |
| **Transcription** | Single `text` field per utterance | List of time-aligned segments |
| **Personal information** | Excluded utterances | Silenced in audio, flagged in segments |
| **Speakers per sample** | One | Three (Speaker A, Speaker B, Recorder) |


## Example Use Cases

- **Conversational ASR**: Train and evaluate ASR models on long-form, multi-speaker Danish audio.
- **Speaker diarization**: Use the speaker-annotated segments for diarization research.
- **Dialogue analysis**: Study turn-taking patterns, conversational dynamics, and Danish spoken language.


## Forbidden Use Cases

Speech synthesis and biometric identification are not allowed using this dataset. For more information, see addition 4 in our [license](https://huggingface.co/datasets/alexandrainst/coral/blob/main/LICENSE).


## License

The dataset is licensed under an OpenRAIL-D license, adapted from OpenRAIL-M, which allows commercial use with a few restrictions (such as speech synthesis and biometric identification). See [license](https://huggingface.co/datasets/alexandrainst/coral/blob/main/LICENSE).


## Creators and Funders

The CoRal project is funded by the [Danish Innovation Fund](https://innovationsfonden.dk/) and consists of the following partners:

- [Alexandra Institute](https://alexandra.dk/)
- [University of Copenhagen](https://www.ku.dk/)
- [Agency for Digital Government](https://digst.dk/)
- [Alvenir](https://www.alvenir.ai/)
- [Corti](https://www.corti.ai/)


## Citation

```bibtex
@dataset{coral2024,
  author    = {Dan Saattrup Smart, Sif Bernstorff Lehmann, Simon Rands Leminen, Anders Jess Pedersen, Anna Katrine van Zee and Torben Blach},
  title     = {CoRal: A Diverse Danish ASR Dataset Covering Dialects, Accents, Genders, and Age Groups},
  year      = {2024},
  url       = {https://hf.co/datasets/alexandrainst/coral},
}
```