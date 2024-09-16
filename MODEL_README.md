# Røst-315m

This is a Danish state-of-the-art speech recognition model, trained by [the Alexandra
Institute](https://alexandra.dk/).


## Quick Start
Start by installing the required libraries:

```shell
$ pip install transformers kenlm pyctcdecode
```

Next you can use the model using the `transformers` Python package as follows:

```python
>>> from transformers import pipeline
>>> audio = get_audio()  # 16kHz raw audio array
>>> transcriber = pipeline(model="alexandrainst/roest-315m")
>>> transcriber(audio)
{'text': 'your transcription'}
```


## Evaluation Results

Mean character error rates on various test sets, compared to other models (lower is
better; best scores in **bold**, second-best in *italics*):

| Model | Number of parameters | [CoRal](https://huggingface.co/datasets/alexandrainst/coral/viewer/read_aloud/test) | [Danish Common Voice 17](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0/viewer/da/test) |
|:---|---:|---:|---:|
| Røst-315m (this model) | 315M | X | X |
| [chcaa/xls-r-300m-danish-nst-cv9](https://hf.co/chcaa/xls-r-300m-danish-nst-cv9) | 315M | X | 4.1% |
| [mhenrichsen/hviske](https://hf.co/mhenrichsen/hviske) | 1540M | X | 5.3% |
| [openai/whisper-large-v3](https://hf.co/openai/whisper-large-v3) | 1540M | X | 7.7% |
| [openai/whisper-large-v2](https://hf.co/openai/whisper-large-v2) | 1540M | X | 10.7% |
| [openai/whisper-large](https://hf.co/openai/whisper-large) | 1540M | X | 12.8% |
| [openai/whisper-medium](https://hf.co/openai/whisper-medium) | 764M | X | 13.2% |
| [openai/whisper-small](https://hf.co/openai/whisper-small) | 242M | X | 22.2% |


### Detailed Evaluation Across Demographics on the CoRal Test Set

| Dialect | CER | WER |
|:---|---:|
| Københavnsk | X | X |
| Sjællandsk | X | X |
| Fynsk | X | X |
| Sønderjysk | X | X |
| Vestjysk | X | X |
| Østjysk | X | X |
| Nordjysk | X | X |
| Sydømål | X | X |
| Bornholmsk | X | X |
| Non-native | X | X |

| Gender | CER | WER |
|:---|---:|
| Female | X | X |
| Male | X | X |

| Age group | CER | WER |
|:---|---:|
| 0-25 | X | X |
| 25-50 | X | X |
| 50+ | X | X |


## Training Data

The base model used,
[`chcaa/xls-r-300m-danish`](https://huggingface.co/chcaa/xls-r-300m-danish), was
pretrained on 141,000 hours of Danish radio (more specifically, DR P1 and Radio24Syv
from 2005 to 2021).

This finetuned model has been further trained on the read-aloud training split of the
[CoRal dataset](https://huggingface.co/datasets/alexandrainst/coral) (revision
0c387d3b6bdfe6e621aa34025505cb893270884b), consisting of approximately 365 hours of
Danish read-aloud speech, diverse across dialects, accents, ages and genders.

An n-gram language model has been trained separately, and is used to guide the
transcription generation of the speech recognition model. This n-gram language model has
been trained on all of the [Danish
Wikipedia](https://huggingface.co/datasets/alexandrainst/scandi-wiki/viewer/da)
(approximately 287,000 articles).


## Intended use cases

This model is intended to be used for Danish automatic speech recognition.

Note that Biometric Identification is not allowed using the CoRal dataset and/or derived
models. For more information, see addition 4 in our
[license](https://huggingface.co/datasets/alexandrainst/roest-315m/blob/main/LICENSE).


## Why the name Røst?

Røst is both the [Danish word for the human
voice](https://ordnet.dk/ddo/ordbog?query=r%C3%B8st) as well as being the name of [one
of the cold-water coral reefs in
Scandinavia](https://da.wikipedia.org/wiki/Koralrev#Koldtvandskoralrev).


## License
The dataset is licensed under a custom license, adapted from OpenRAIL-M, which allows
commercial use with a few restrictions (speech synthesis and biometric identification).
See
[license](https://huggingface.co/datasets/alexandrainst/roest-315m/blob/main/LICENSE).


## Creators and Funders
The CoRal project is funded by the [Danish Innovation
Fund](https://innovationsfonden.dk/) and consists of the following partners:

- [Alexandra Institute](https://alexandra.dk/)
- [University of Copenhagen](https://www.ku.dk/)
- [Agency for Digital Government](https://digst.dk/)
- [Alvenir](https://www.alvenir.ai/)
- [Corti](https://www.corti.ai/)


## Citation

We will submit a research paper soon, but until then, if you use this model in your
research or development, please cite it as follows:

```bibtex
@dataset{coral2024,
  author    = {Dan Saattrup Nielsen, Sif Bernstorff Lehmann, Simon Leminen Madsen, Anders Jess Pedersen, Anna Katrine van Zee and Torben Blach},
  title     = {CoRal: A Diverse Danish ASR Dataset Covering Dialects, Accents, Genders, and Age Groups},
  year      = {2024},
  url       = {https://hf.co/datasets/alexandrainst/coral},
}
```
