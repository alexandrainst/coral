# CoRal: Danish Conversational and Read-aloud Dataset

## Dataset Overview

**CoRal** is a comprehensive Automatic Speech Recognition (ASR) dataset designed to
capture the  diversity of the Danish language across various dialects, accents, genders,
and age groups. The primary goal of the CoRal dataset is to provide a robust resource
for training and evaluating ASR models that can understand and transcribe spoken Danish
in all its variations.

### Key Features

- **Dialect and Accent Diversity**: The dataset includes speech samples from all major
  Danish dialects as well as multiple accents, ensuring broad geographical coverage and
  the inclusion of regional linguistic features.
- **Gender Representation**: Both male and female speakers are well-represented,
  offering balanced gender diversity.
- **Age Range**: The dataset includes speakers from a wide range of age groups,
  providing a comprehensive resource for age-agnostic ASR model development.
- **High-Quality Audio**: All recordings are of high quality, ensuring that the dataset
  can be used for both training and evaluation of high-performance ASR models.

### Quick Start

The CoRal dataset is ideal for training ASR models that need to generalise across
different dialects and speaker demographics within the Danish language. Below is an
example of how to load and use the dataset with Hugging Face's `datasets` library:

```python
from datasets import load_dataset

# Load the Coral dataset
coral = load_dataset("alexandrainst/coral", "read_aloud")

# Example: Accessing an audio sample and its transcription
sample = coral['train'][0]
audio = sample['audio']
transcription = sample['text']

print(f"Audio: {audio['array']}")
print(f"Text: {transcription}")
```


## Data Fields

- `id_recording`: Unique identifier for the recording.
- `id_sentence`: Unique identifier for the text being read aloud.
- `id_speaker`: Unique identifier for each speaker.
- `text`: Text being read aloud.
- `location`: Address of recording place.
- `location_roomdim`: Dimension of recording room.
- `noise_level`: Noise level in the room, given in dB.
- `noise_type`: Noise exposed to the speaker while recording. Note the noise is not
  present in the audio, but is there to mimic differences in speech in a noisy
  environment.
- `source_url`: URL to the source of the text.
- `age`: Age of the speaker.
- `gender`: Gender of the speaker.
- `dialect`: Self-reported dialect of the speaker.
- `country_birth`: Country where the speaker was born.
- `validated`: Manual validation state of the recording.
- `audio`: The audio file of the recording.
- `asr_prediction`: ASR output prediction of the `asr_validation_model`.
- `asr_validation_model`: Hugging Face Model ID used for automatic validation of the
  recordings.
- `asr_wer`: Word error rate between `asr_prediction` and `text`.
- `asr_cer`: Character error rate between `asr_prediction` and `text`.


## Read-aloud Data Statistics

### Train Split

There are ~365 hours of audio in the train split, with ~565 speakers, reading ~152,000
unique sentences aloud.

Gender distribution:
- female: 74%
- male: 26%

Dialect distribution:
- Bornholmsk: 2%
- Fynsk: 5%
- Københavnsk: 15%
- Nordjysk: 16%
- Sjællandsk: 14%
- Sydømål: 0.2%
- Sønderjysk: 5%
- Vestjysk: 11%
- Østjysk: 33%

Accent distribution:
- native: 97%
- foreign: 3%

Age group distribution:
- 0-24: 6%
- 25-49: 36%
- 50-: 57%

### Validation Split

There are 2.97 hours of audio in the validation split, with 10 speakers, reading 1,789
unique sentences aloud.

Gender distribution:
- female: 45%
- male: 55%

Dialect distribution:
- Bornholmsk: 7%
- Fynsk: 11%
- Københavnsk: 9%
- Nordjysk: 8%
- Sjællandsk: 12%
- Sydømål: 23%
- Sønderjysk: 13%
- Vestjysk: 8%
- Østjysk: 9%

Accent distribution:
- native: 81%
- foreign: 19%

Age group distribution:
- 0-24: 11%
- 25-49: 39%
- 50-: 51%

### Test Split

There are 8.44 hours of audio in the test split, with 22 speakers, reading 5,287 unique
sentences aloud.

Gender distribution:
  - female: 58%
  - male: 42%

Dialect distribution:
  - Bornholmsk: 10%
  - Fynsk: 10%
  - Københavnsk: 10%
  - Nordjysk: 11%
  - Sjællandsk: 11%
  - Sydømål: 14%
  - Sønderjysk: 10%
  - Vestjysk: 12%
  - Østjysk: 11%

Accent distribution:
- native: 90%
- foreign: 10%

Age group distribution:
  - 0-24: 28%
  - 25-49: 32%
  - 50-: 40%


## Conversational Data Statistics

The conversational data is not yet available, but we are working on it and plan to
release it during 2024.


## Example Use Cases

### ASR Model Training

Train robust ASR models that can handle dialectal variations and diverse speaker
demographics in Danish.

### Dialect Studies

Analyse the linguistic features of different Danish dialects.

### Forbidden Use Cases

Speech Synthesis and Biometric Identification are not allowed using the CoRal dataset.
For more information, see addition 4 in our
[license](https://huggingface.co/datasets/alexandrainst/coral/blob/main/LICENSE).


## License
The dataset is licensed under a custom license, adapted from OpenRAIL-M, which allows
commercial use with a few restrictions (speech synthesis and biometric identification).
See [license](https://huggingface.co/datasets/alexandrainst/coral/blob/main/LICENSE).


## Creators and Funders
The CoRal project is funded by the [Danish Innovation
Fund](https://innovationsfonden.dk/) and consists of the following partners:

- [Alexandra Institute](https://alexandra.dk/)
- [University of Copenhagen](https://www.ku.dk/)
- [Agency for Digital Government](https://digst.dk/)
- [Alvenir](https://www.alvenir.ai/)
- [Corti](https://www.corti.ai/)


## Citation
We will submit a research paper soon, but until then, if you use the CoRal dataset in
your research or development, please cite it as follows:

```bibtex
@dataset{coral2024,
  author    = {Sif Bernstorff Lehmann, Dan Saattrup Nielsen, Simon Leminen Madsen, Anders Jess Pedersen, Anna Katrine van Zee and Torben Blach},
  title     = {CoRal: A Diverse Danish ASR Dataset Covering Dialects, Accents, Genders, and Age Groups},
  year      = {2024},
  url       = {https://hf.co/datasets/alexandrainst/coral},
}
```
