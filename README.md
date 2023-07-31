# CoRal Models

Danish ASR and TTS models associated with the CoRal project.

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/CoRal-models/coral_models.html)
[![License](https://img.shields.io/github/license/alexandrainst/CoRal-models)](https://github.com/alexandrainst/CoRal-models/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/CoRal-models)](https://github.com/alexandrainst/CoRal-models/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-53%25-orange.svg)](https://github.com/alexandrainst/CoRal-models/tree/main/tests)


Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)


## Setup

### Set up the environment

1. Run `make install`, which installs Poetry (if it isn't already installed), sets up a virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

If you're on MacOS and get an error saying something along the lines of "fatal error:
'lzma.h' file not found" then try the following and rerun `make install` afterwards:

```
export CPPFLAGS="-I$(brew --prefix)/include"
```


### Install new packages

To install new PyPI packages, run:

```
$ poetry add <package-name>
```

### Get an overview of the available commands

Simply write `make` to display a list of the commands available. This includes the
above-mentioned `make install` command, as well as building and viewing documentation,
publishing the code as a package and more.


## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management
* [hydra](https://hydra.cc/): Manage configuration files
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project


## Project structure
```
.
├── .flake8
├── .github
│   └── workflows
│       ├── ci.yaml
│       └── docs.yaml
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── config
│   ├── __init__.py
│   ├── config.yaml
│   ├── dataset
│   │   ├── common_voice_da.yaml
│   │   ├── ftspeech.yaml
│   │   └── test.yaml
│   ├── hydra
│   │   └── job_logging
│   │       └── custom.yaml
│   └── model
│       ├── test.yaml
│       ├── wav2vec2.yaml
│       ├── wav2vec2_with_lm.yaml
│       └── whisper.yaml
├── data
├── makefile
├── models
├── notebooks
├── poetry.toml
├── pyproject.toml
├── src
│   ├── coral_models
│   │   ├── __init__.py
│   │   ├── compute_metrics.py
│   │   ├── data.py
│   │   ├── finetune.py
│   │   ├── protocols.py
│   │   ├── utils.py
│   │   └── wav2vec2.py
│   └── scripts
│       ├── build_ftspeech.py
│       ├── evaluate.py
│       ├── finetune.py
│       ├── fix_dot_env_file.py
│       ├── push_ftspeech_to_hub.py
│       ├── train_ngram_decoder.py
│       └── versioning.py
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── test_compute_metrics.py
    ├── test_data.py
    ├── test_finetune.py
    ├── test_protocols.py
    ├── test_utils.py
    └── test_wav2vec2.py
```
