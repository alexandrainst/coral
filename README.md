# CoRal Models

Danish ASR and TTS models associated with the CoRal project.

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/CoRal-models/coral_models.html)
[![License](https://img.shields.io/github/license/alexandrainst/CoRal-models)](https://github.com/alexandrainst/CoRal-models/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/CoRal-models)](https://github.com/alexandrainst/CoRal-models/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/alexandrainst/CoRal-models/tree/main/tests)


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
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── config
│   ├── __init__.py
│   └── config.yaml
├── data
├── makefile
├── models
├── notebooks
├── poetry.toml
├── pyproject.toml
├── src
│   ├── scripts
│   │   ├── fix_dot_env_file.py
│   │   └── versioning.py
│   └── coral_models
│       └── __init__.py
└── tests
    └── __init__.py
```
