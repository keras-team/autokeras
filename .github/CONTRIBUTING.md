# Contributing Guide

Contributions are welcome, and greatly appreciated!
Follow the tag of [good first issue](https://github.com/keras-team/autokeras/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
for the issues for beginner.

## Pull Request Guide

1. Is this the first pull request that you're making with GitHub? If so, read the guide [Making a pull request to an open-source project](https://github.com/gabrieldemarmiesse/getting_started_open_source).

2. Include "resolves #issue_number" in the description of the pull request if applicable. Briefly describe your contribution.

3. Submit the pull request from the first day of your development and create it as a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/). Click `ready for review` when finished and passed the all the checks.

4. For the case of bug fixes, add new test cases which would fail before your bug fix.


## Setup Environment

### virtualenv
Install [Virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).
Create a new virtualenv named `ak` based on python3.
```
mkvirtualenv -p python3 ak 
```
Please use this virtualenv for development.

### Installation
Clone the repo. Go to the repo directory.
Run the following commands.
```
workon ak

pip install -e ".[tests]"
pip uninstall autokeras
add2virtualenv .

pip install mkdocs
pip install mkdocs-material
pip install autopep8
pip install git+https://github.com/gabrieldemarmiesse/typed_api.git@0.1.1
``` 


## Run Tests

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

## Code Style Guide
1. Run `shell/format.sh` to format your code.
2. Run `shell/lint.sh` to check.
3. Docstrings should follow our style.

For "our style", just check the code of AutoKeras.

### pre-commit hook
Check code style automatically at every commit. 
Go to the repo directory.
```
cp shell/lint.sh .git/hooks/pre-commit
```
If the check doesn't pass, the commit will be aborted.

