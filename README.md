# CoRal

Danish ASR and TTS datasets and models, as part of the [CoRal
project](https://alexandra.dk/coral/), funded by the [Innovation
Fund](https://innovationsfonden.dk/).

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/coral/coral.html)
[![License](https://img.shields.io/github/license/CoRal-project/coral)](https://github.com/CoRal-project/coral/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/CoRal-project/coral)](https://github.com/CoRal-project/coral/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-57%25-orange.svg)](https://github.com/CoRal-project/coral/tree/main/tests)


Author and maintainer:

- Dan Saattrup Smart (dan.smart@alexandra.dk)


## Installation

1. Run `make install`, which installs `uv` (if it isn't already installed), sets up a
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
  - `coral_read_aloud`
  - `coral_conversation`
  - `coral_tts`
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
- `enable_experiment_tracking`: Whether training monitoring during training should be
  enabled. Defaults to false. You can also set `experiment_tracking` to either `wandb`
  or `mlflow` to specify which experiment tracking tool to use (`wandb` is used by
  default).
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
  Hugging Face Hub. Note that subsets are separated with "::". Defaults to
  `CoRal-project/coral_v3::conversation`.
- `eval_split_name`: The dataset split to evaluate on. Defaults to `test`.
- `text_column`: The name of the column in the dataset that contains the text. Defaults
  to `text`.
- `audio_column`: The name of the column in the dataset that contains the audio. Defaults
  to `audio`.

See all the evaluation options in the `config/evaluation.yaml` file.


## Troubleshooting

If you're on MacOS and get an error saying something along the lines of "fatal error:
'lzma.h' file not found" then try the following and rerun `make install` afterwards:

```
export CPPFLAGS="-I$(brew --prefix)/include"
```

Another MacOS issue can happen if you get something like "fatal error: 'cstddef' file
not found" and/or "fatal error: 'climits' file not found". In this case, first ensure
that [you have Homebrew installed](https://brew.sh/), after which you run the following:

```
brew install cmake boost zlib eigen
```
