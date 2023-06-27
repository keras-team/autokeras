# Contributing Guide

Contributions are welcome, and greatly appreciated!
This page is only a guide of the best practices of contributing code to AutoKeras.
The best way to contribute is to join our community by reading [this](https://autokeras.com/#contributing-code).
We will get you started right away.

Follow the tag of [good first issue](https://github.com/keras-team/autokeras/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
for the issues for beginner.

## Pull Request Guide

1. Is this the first pull request that you're making with GitHub? If so, read the guide [Making a pull request to an open-source project](https://github.com/gabrieldemarmiesse/getting_started_open_source).

2. Include "resolves #issue_number" in the description of the pull request if applicable. Briefly describe your contribution.

3. Submit the pull request from the first day of your development and create it as a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/). Click `ready for review` when finished and passed the all the checks.

4. For the case of bug fixes, add new test cases which would fail before your bug fix.


## Setup Environment
We introduce 3 different options: **GitHub Codespaces**, **VS Code & Dev Containers**, **the general setup**.
You can choose base on your preference.

### Option 1: GitHub Codespaces
You can simply open the repository in GitHub Codespaces.
The environment is already setup there.

### Option 2: VS Code & Dev Containers
Open VS Code.
Install the `Dev Containers` extension.
Press `F1` key. Enter `Dev Containers: Open Folder in Container...` to open the repository root folder.
The environment is already setup there.

### Option 3: The General Setup

Install [Virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).
Create a new virtualenv named `ak` based on python3.
```
mkvirtualenv -p python3 ak 
```
Please use this virtualenv for development.

Clone the repo. Go to the repo directory.
Run the following commands.
```
workon ak

pip install -e ".[tests]"
pip uninstall autokeras
add2virtualenv .
``` 

## Run Tests

### GitHub Codespaces or VS Code & Dev Containers
If you are using "GitHub Codespaces" or "VS Code & Dev Containers",
you can simply open any `*_test.py` file under the `tests` directory,
and wait a few seconds, you will see the test tab on the left of the window.

### General Setup

If you are using the general setup.

Activate the virtualenv.
Go to the repo directory
Run the following lines to run the tests.

Run all the tests.
```
pytest tests
```

Run all the unit tests.
```
pytest tests/autokeras
```

Run all the integration tests.
```
pytest tests/integration_tests
```

## Code Style
You can run the following manually every time you want to format your code.
1. Run `shell/format.sh` to format your code.
2. Run `shell/lint.sh` to check.

## Docstrings
Docstrings should follow our style.
Just check the style of other docstrings in AutoKeras.
