# Contributing Guide

Contributions are welcome, and greatly appreciated!
We recommend you to check our [Developer Tools Guide](#developer-tools-guide)
to make the development process easier and standard.

Notably, you can follow the tag of [call for contributors](https://github.com/keras-team/autokeras/labels/call%20for%20contributors) in the issues.
Those issues are designed for the external contributors to solve.
The easiest ones are tagged with [good first issue](https://github.com/keras-team/autokeras/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
The pull requests solving these issues are most likely to be merged.

## Pull Request Guide
Before you submit a pull request, check that it meets these guidelines:

1. Is this the first pull request that you're making with GitHub? If so, read the guide [Making a pull request to an open-source project](https://github.com/gabrieldemarmiesse/getting_started_open_source).

2. Include "resolves #issue_number" in the description of the pull request if applicable. Briefly describe your contribution.

3. Submit the pull request from the first day of your development and create it as a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/). Click `ready for review` when finished and passed the all the checks.

4. For the case of bug fixes, add new test cases which would fail before your bug fix.


## Setup your environment

### virtualenv
Install [Virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).
Create a new virtualenv based on python3.6+.

### Installation
Clone the repo. Go to the repo directory. Activate the virtualenv. Run the following commands.
```
pip install -e ".[tests]"
pip uninstall autokeras
add2virtualenv .
pip install mkdocs
pip install mkdocs-material
pip install isort
pip install autopep8
``` 

### pre-commit hook
You can make git run `flake8` before every commit automatically. It will make you go faster by
avoiding pushing commit which are not passing the flake8 tests. To do this,
open `.git/hooks/pre-commit` with a text editor and write `flake8` inside. If the `flake8` doesn't
pass, the commit will be aborted.

## Code Style Guide
1. Run `shell/format.sh` to format your code.
2. Run `flake8` to check.
3. Docstrings should follow our style.

For "our style", just check the code of AutoKeras.
