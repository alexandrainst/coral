# CoRal Models

Danish ASR and TTS models associated with the CoRal project.

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/coral/coral.html)
[![License](https://img.shields.io/github/license/alexandrainst/coral)](https://github.com/alexandrainst/coral/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/coral)](https://github.com/alexandrainst/coral/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-60%25-yellow.svg)](https://github.com/alexandrainst/coral/tree/main/tests)


Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)


## Quickstart

1. Run `make install`, which installs Poetry (if it isn't already installed), sets up a
   virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.
3. Run `make` to see a list of available commands.

If you're on MacOS and get an error saying something along the lines of "fatal error:
'lzma.h' file not found" then try the following and rerun `make install` afterwards:

```
export CPPFLAGS="-I$(brew --prefix)/include"
```
