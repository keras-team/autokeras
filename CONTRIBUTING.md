# Contributing Guide

Contributions are welcome, and greatly appreciated! Every little bit helps, and credit will always be given.
We recommend you to check our [Developer Tools Guide](#developer-tools-guide) 
to make the development process easier and standard.

Notably, you can follow the tag of [call for contributors](https://github.com/keras-team/autokeras/labels/call%20for%20contributors) in the issues.
Those issues are designed for the external contributors to solve.
The pull requests solving these issues are most likely to be merged.

There are many ways to contribute to Auto-Keras,
including submit feedback, fix bugs, implement features, and write documentation.
The guide for each type of contribution is as follows.

#### Submit Feedback
The feedback should be submitted by creating an issue at [GitHub issues](https://github.com/keras-team/autokeras/issues).
Select the related template (bug report, feature request, or custom) and add the corresponding labels.

#### Fix Bugs:
You may look through the [GitHub issues](https://github.com/keras-team/autokeras/issues) for bugs.
Anything tagged with "bug report" is open to whoever wants to implement it.
Please follow the 
[Pull Request Guide](#pull-request-guide) to submit your pull request. 
Please also read
[Code Style Guide](#code-style-guide),
and
[Documentation Guide](#documentation-guide)
to ensure your merge request meet our requirements.

#### Implement Features
You may look through the [GitHub issues](https://github.com/keras-team/autokeras/issues) for feature requests.
Anything tagged with "feature request" is open to whoever wants to implement it.
Please follow the 
[Pull Request Guide](#pull-request-guide) to submit your pull request. 
Please also read
[Code Style Guide](#code-style-guide),
[Documentation Guide](#documentation-guide),
and
[Testing Guide](https://github.com/keras-team/autokeras/blob/master/CONTRIBUTING.md#testing-guide)
to ensure your merge request meet our requirements.

#### Write Documentation
The documentation of Auto-Keras is either directly written into the Markdown files in
[mkdocs directory](https://github.com/keras-team/autokeras/tree/master/mkdocs/docs),
or automatically extracted from the docstrings by executing the [autogen.py](https://github.com/keras-team/autokeras/blob/master/mkdocs/autogen.py).
In the first situation, you only need to change the markdown file.
In the second situation, you need to change the docstrings and execute [autogen.py](https://github.com/keras-team/autokeras/blob/master/mkdocs/autogen.py) to update the Markdown files.
Please follow the 
[Pull Request Guide](#pull-request-guide) to submit your pull request. 
Please also read
[Documentation Guide](#documentation-guide)
to ensure your merge request meet our requirements.

## Pull Request Guide
Before you submit a pull request, check that it meets these guidelines:

1. Fork the repository. Create a new branch from the master branch. Give your new branch a **meaningful** name.

2. Pull request from your new branch to the master branch of the original autokeras repo. Give your pull request a **meaningful** name.

3. Include "resolves #issue_number" in the description of the pull request and briefly describe your contribution.

4. Submit the pull request from the first day of your development (after your first commit) and prefix the title of the pull request with **[WIP]**.

5. When the contribution is complete, make sure the pull request passed the CI tests. Change the **[WIP]** to **[MRG]**.
Set the reviewer to 
[@jhfjhfj1](https://github.com/jhfjhfj1).

6. For the case of bug fixes, add new test cases which would fail before your bug fix.

7. If you are a collaborator of the autokeras repo, you don't need to fork the repository. Just create a new branch directly. You also need to change the assignee to the reviewer when request for code review. The reviewer will change the assignee back to you when finished the review. The assignee always means who should push the progress of the pull request now.


## Code Style Guide
This project tries to closely follow the official Python Style Guide detailed in [PEP8](https://www.python.org/dev/peps/pep-0008/).
The docstrings follow the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#381-docstrings).
Please follow these style guide closely, especially for the docstrings,
which would be extracted automatically to generate the documentation.

## Documentation Guide:
The documentation should be provided in two ways, docstring, tutorial, and readme file.
We prefer the documentation to be as complete as possible.

### Docstring
All the methods and classes may directly be called by the user need to be documented with docstrings.
The docstrings should contain all the fields required by the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#381-docstrings).

### Tutorial
You only need to add tutorials to your code if you are contributing or updating a new task module,
e.g. TextClassifier, VideoClassifier,
or a new function could be directly called by the user.
You can modify `mkdocs/docs/start.md` to add your tutorial.
The code example of your new task module should be added to the `examples` directory.

### Readme File
You only need to add tutorials to your code if you are contributing or updating a new task module,
e.g. TextClassifier, VideoClassifier.
The readme file should be named as `README.md`.
It should be written in Markdown.
The content should contain your name, affiliation, and any reference to the method you use.

## Testing Guide
[Pytest](https://docs.pytest.org/en/latest/) is used to write the unit tests of Auto-Keras.
You should test your code by writing unit testing code in `tests` directory.
The testing file name should be the `.py` file with a prefix of `test_` in the corresponding directory,
e.g., the name should be `test_layers.py` if the code of which is to test `layer.py`.
The tests should be run in the root directory of the project by executing the `cov.sh` file.
It would output the coverage information into a directory named `htmlcov`.
Please make sure the code coverage percentage does not decrease after your contribution,
otherwise, the code will not be merged.

## Developer Tools Guide
We highly recommend you to use [Pycharm](https://www.jetbrains.com/pycharm/) 
and [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).
### Pycharm
Pycharm is the best IDE for large project development in Python.
We recommend you [inspect the code](https://www.jetbrains.com/help/pycharm/running-inspections.html)
before you pull request to fix any error and warning suggested by the inspection.
### Virtualenvwrapper
Virtualenvwrapper is a tool to build separated Python environment for each project.
In this way, you can install a different version of Tensorflow, Pytorch, or any other package for each project.
We recommend you to create a virtualenv for autokeras development with virtualenvwrapper,
and only install the packages required by autokeras with the corresponding version.
The virtualenv should be created based on Python 3.6 interpreter.
Use pycharm to select the 
[virtualenv as interpreter](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html).

## Reusable Code Guide
You may checkout this code review [video](https://youtu.be/PWdfY0DmjCo) to get familiar with the code structure.
Other than the base classes you have to extend,
there are some other classes you can extend.

### ModelTrainer
[`autokeras.model_trainer.ModelTrainer`](https://github.com/keras-team/autokeras/blob/master/autokeras/model_trainer.py) is a class for training Pytorch models.
If needed a new metric or loss function other than the ones we have, you can add your own to [`loss_function.py`](https://github.com/keras-team/autokeras/blob/master/autokeras/loss_function.py) and [`metric.py`](https://github.com/keras-team/autokeras/blob/master/autokeras/metric.py).
You can follow its [documentation](https://github.com/keras-team/autokeras/blob/master/autokeras/model_trainer.py) and this [example](https://github.com/keras-team/autokeras/blob/master/examples/code_reuse_example.py) to use it.
Make sure your loss function, metric, Pytorch model, and Dataloader are compatible with each other.

## Main Contributor List
We really appreciate all the contributions.
To show our appreciation to those who contributed most,
we would like to maintain a list of main contributors.
To be in the list, you need to meet the following requirments.
1. Be on campus of Texas A&M University.
2. Constantly present in our meetings.
3. Constantly contribute code to our repository.
4. Keep the above for over 6 months.
