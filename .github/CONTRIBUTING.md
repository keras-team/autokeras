# Contributing Guide

Contributions are welcome, and greatly appreciated!
We recommend you to check our [Developer Tools Guide](#developer-tools-guide)
to make the development process easier and standard.

Notably, you can follow the tag of [call for contributors](https://github.com/keras-team/autokeras/labels/call%20for%20contributors) in the issues.
Those issues are designed for the external contributors to solve.
The pull requests solving these issues are most likely to be merged.

There are many ways to contribute to AutoKeras,
including submit feedback, fix bugs, implement features, and write documentation.
The guide for each type of contribution is as follows.

#### Submit Feedback
The feedback should be submitted by creating an issue at [GitHub issues](https://github.com/keras-team/autokeras/issues).
Select the related template (bug report, feature request, or custom) to write the details of the feedback.

#### Fix Bugs:
You may look through the [GitHub issues](https://github.com/keras-team/autokeras/issues) for bugs.
Anything tagged with "bug report" is open to whomever wants to fix.
Please follow the
[Pull Request Guide](#pull-request-guide) to submit your pull request.
Please also read
[Code Style Guide](#code-style-guide),
to ensure your pull request meet our requirements.

#### Implement Features
You may look through the [GitHub issues](https://github.com/keras-team/autokeras/issues) for feature requests.
Anything tagged with "feature request" is open to whoever wants to implement it.
Unit tests are needed to test the new feature implemented and fully cover the new code you wrote.
Please follow the
[Pull Request Guide](#pull-request-guide) to submit your pull request.
Please also read
[Code Style Guide](#code-style-guide),
to ensure your pull request meet our requirements.

#### Write Documentation
The documentation of AutoKeras is either directly written into the Markdown files in
[templates directory](https://github.com/keras-team/autokeras/tree/master/docs/templates),
or automatically extracted from the docstrings by executing the [autogen.py](https://github.com/keras-team/autokeras/blob/master/docs/autogen.py).
Please follow the
[Pull Request Guide](#pull-request-guide) to submit your pull request.
Please also read
[Code Style Guide](#code-style-guide).

## Pull Request Guide
Before you submit a pull request, check that it meets these guidelines:

1. Fork the repository. Create a new branch from the master branch. Give your new branch a **meaningful** name.

2. Pull request from your new branch to the master branch of the original autokeras repo. Give your pull request a **meaningful** name.

3. Include "resolves #issue_number" in the description of the pull request if applicable and briefly describe your contribution.

4. Submit the pull request from the first day of your development and create it as a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/). Click `ready for review` when finished and passed the all the checks.

6. For the case of bug fixes, add new test cases which would fail before your bug fix.

## Code Style Guide
1. Your code should pass the `flake8` check.
2. Docstrings should follow our style.
3. Imports should follow our style.
For "our style", just check the code of AutoKeras.

## Developer Tools Guide
We highly recommend you to use [Pycharm](https://www.jetbrains.com/pycharm/)
and [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)
as well as a pre-commit hook.

### Visual Studio Code
We suggest using Visual Studio Code (VSCode) to develop for AutoKeras. It can automatically format your code, sort your imports, mark any code style violations, generate docstrings according to our format for you. Check VSCode configuration [here](https://gist.github.com/jhfjhfj1/68714194e2faa6fb81c53eea59779458).

### virtualenv
Use [virtualenv](https://virtualenv.pypa.io/en/latest/) to create a isolated Python environment for AutoKeras project to avoid dependency version conflicts with your other projects.
[Virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) is a even simpler command-line tool to manage multiple virtualenvs.

### pre-commit hook
You can make git run `flake8` before every commit automatically. It will make you go faster by
avoiding pushing commit which are not passing the flake8 tests. To do this,
open `.git/hooks/pre-commit` with a text editor and write `flake8` inside. If the `flake8` doesn't
pass, the commit will be aborted.

## Main Contributor List
We really appreciate all the contributions.
To show our appreciation to those who contributed most,
we would like to maintain a list of main contributors.
To be in the list, you need to meet the following requirments.
1. Be on campus of Texas A&M University.
2. Constantly present in our meetings.
3. Constantly contribute code to our repository.
4. Keep the above for over 6 months.
