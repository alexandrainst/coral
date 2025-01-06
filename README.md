# CoRal

Danish ASR and TTS datasets and models, as part of the [CoRal
project](https://alexandra.dk/coral/), funded by the [Innovation
Fund](https://innovationsfonden.dk/).

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/coral/coral.html)
[![License](https://img.shields.io/github/license/alexandrainst/coral)](https://github.com/alexandrainst/coral/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/coral)](https://github.com/alexandrainst/coral/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-54%25-orange.svg)](https://github.com/alexandrainst/coral/tree/main/tests)


Developers:

- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)
- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
- Simon Leminen Madsen (simon.leminen@alexandra.dk)



## Installation

1. Run `make install`, which installs Poetry (if it isn't already installed), sets up a
   virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.
3. Run `make` to see a list of available commands.


## Usage

### Finetuning an Acoustic Model for Automatic Speech Recognition (ASR)

You can use the `finetune_asr_model` script to finetune your own ASR model:

```bash
python src/scripts/finetune_asr_model.py [key=value]...
```

Here are some of the more important available keys:

- `model`: The base model to finetune. Supports the following values:
  - `wav2vec2-small`
  - `wav2vec2-medium`
  - `wav2vec2-large`
  - `whisper-xxsmall`
  - `whisper-xsmall`
  - `whisper-small`
  - `whisper-medium`
  - `whisper-large`
  - `whisper-large-turbo`
- `datasets`: The datasets to finetune the models on. Can be a single dataset or an
  array of datasets (written like [dataset1,dataset2,...]). Supports the following
  values:
  - `coral`
  - `common_voice_17`
  - `common_voice_9`
  - `fleurs`
  - `ftspeech`
  - `nota`
  - `nst`
- `dataset_probabilities`: In case you are finetuning on several datasets, you need to
  specify the probability of sampling each one. This is an array of probabilities that
  need to sum to 1. If not set, the datasets are sampled uniformly.
- `model_id`: The model ID of the finetuned model. Defaults to the model type along with
  a timestamp.
- `push_to_hub`, `hub_organisation` and `private`: Whether to push the finetuned model
  to the Hugging Face Hub, and if so, which organisation to push it to. If `private` is
  set to `True`, the model will be private. The default is not to push the model to the
  Hub.
- `wandb`: Whether Weights and Biases should be used for monitoring during training.
  Defaults to false.
- `per_device_batch_size` and `dataloader_num_workers`: The batch size and number of
  workers to use for training. Defaults to 8 and 4, respectively. Tweak these if you are
  running out of GPU memory.
- `model.learning_rate`, `total_batch_size`, `max_steps`, `warmup_steps`: Training
  parameters that you can tweak, although it shouldn't really be needed.

See all the finetuning options in the `config/asr_finetuning.yaml` file.


### Evaluating an Automatic Speech Recognition (ASR) Model

You can use the `evaluate_model` script to evaluate an ASR model:

```bash
python src/scripts/evaluate_model.py [key=value]...
```

Here are some of the more important available keys:

- `model_id` (required): The Hugging Face model ID of the ASR model to evaluate.
- `dataset`: The ASR dataset to evaluate the model on. Can be any ASR dataset on the
  Hugging Face Hub. Note that subsets are separated with "::". For instance, to evaluate
  on the Danish Common Voice 17 dataset, you would use
  `mozilla-foundation/common_voice_17_0::da`. Defaults to
  `alexandrainst/coral::read_aloud`.
- `eval_split_name`: The dataset split to evaluate on. Defaults to `test`.
- `text_column`: The name of the column in the dataset that contains the text. Defaults
  to `text`.
- `audio_column`: The name of the column in the dataset that contains the audio. Defaults
  to `audio`.
- `detailed`: Only relevant if evaluating on the (default) CoRal test dataset. This will
  give a detailed evaluation across the different demographics in the dataset. If set to
  False it will only give the overall scores. Defaults to True.

See all the evaluation options in the `config/evaluation.yaml` file.

You can produce a comparison plot of different models evaluated on the CoRal test
dataset with `detailed=True` by running the following script:

```bash
python src/scripts/create_comparison_plot.py \
  -f EVALUATION_FILE [-f EVALUATION_FILE ...] [--metric METRIC]
```

Here the `EVALUATION_FILE` arguments are the paths to the evaluation files produced by
`evaluate_model.py` (they end in `-coral-scores.csv`). The `METRIC` argument is the
metric to compare on, which can be one of `wer` and `cer`, for the word error rate and
character error rate, respectively. The default is `cer`.


## Troubleshooting

If you're on MacOS and get an error saying something along the lines of "fatal error:
'lzma.h' file not found" then try the following and rerun `make install` afterwards:

```
export CPPFLAGS="-I$(brew --prefix)/include"
```
