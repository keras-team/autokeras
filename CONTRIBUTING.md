# Contributing Guide

Contributions are welcome, and greatly appreciated! Every little bit helps, and credit will always be given.
We recommend you to check our [Developer Tools Guide](#developer-tools-guide) 
to make the development process easier and standard.
**The type of contribution we would be most happy to see is new task modules, e.g. TextClassifier, VideoClassifier.**

## Implement New Task Modules
A task module is a comparatively separate module which can handle a specify task.
For example, ImageClassifier is the only task module we have for now.
The list of task modules we are seeking is all the issues with label "new task module" (in progress).

The new task module should be submitted by pull request from the first day you start to develop the module.
Make sure your pull request follow the [Pull Request Guideline](#pull-request-guide).
Your module will be run against our reserved datasets to test the performance.
If your module's performance meets our expectation,
the new task module will be merged into Auto-Keras package.

In general, all new task module should inherit their objects from the `Classifier` class in [`autokeras/classifier.py`](https://github.com/jhfjhfj1/autokeras/blob/master/autokeras/classifier.py).
Reach out to us if you feel there is a special requirement.
For every new feature, a new directory should be created inside the /autokeras directory, e.g. text_classifier.
All the code contributed should be within the directory.
The details of the functions to inherit is in the documentation of [`autokeras/classifier.py`](https://github.com/jhfjhfj1/autokeras/blob/master/autokeras/classifier.py)

Please also read
[Code Style Guide](#code-style-guide),
[Documentation Guide](#documentation-guide),
and
[Testing Guide](https://github.com/jhfjhfj1/autokeras/blob/master/CONTRIBUTING.md#testing-guide)
to ensure your merge request meet our requirements.

## Other Contributions
There are many other ways to contribute to Auto-Keras,
including submit feedback, fix bugs, implement features, and write documentation.
The guide for each type of contribution is as follows.

#### Submit Feedback
The feedback should be submitted by creating an issue at [GitHub issues](https://github.com/jhfjhfj1/autokeras/issues).
Select the related template (bug report, feature request, or custom) and add the corresponding labels.

#### Fix Bugs:
You may look through the [GitHub issues](https://github.com/jhfjhfj1/autokeras/issues) for bugs.
Anything tagged with "bug report" is open to whoever wants to implement it.
Please follow the 
[Pull Request Guide](#pull-request-guide) to submit your pull request. 
Please also read
[Code Style Guide](#code-style-guide),
and
[Documentation Guide](#documentation-guide)
to ensure your merge request meet our requirements.

#### Implement Features
You may look through the [GitHub issues](https://github.com/jhfjhfj1/autokeras/issues) for feature requests.
Anything tagged with "feature request" is open to whoever wants to implement it.
Please follow the 
[Pull Request Guide](#pull-request-guide) to submit your pull request. 
Please also read
[Code Style Guide](#code-style-guide),
[Documentation Guide](#documentation-guide),
and
[Testing Guide](https://github.com/jhfjhfj1/autokeras/blob/master/CONTRIBUTING.md#testing-guide)
to ensure your merge request meet our requirements.

#### Write Documentation
The documentation of Auto-Keras is either directly written into the Markdown files in
[mkdocs directory](https://github.com/jhfjhfj1/autokeras/tree/master/mkdocs/docs),
or automatically extracted from the docstrings by executing the [autogen.py](https://github.com/jhfjhfj1/autokeras/blob/master/mkdocs/autogen.py).
In the first situation, you only need to change the markdown file.
In the second situation, you need to change the docstrings and execute [autogen.py](https://github.com/jhfjhfj1/autokeras/blob/master/mkdocs/autogen.py) to update the Markdown files.
Please follow the 
[Pull Request Guide](#pull-request-guide) to submit your pull request. 
Please also read
[Documentation Guide](#documentation-guide)
to ensure your merge request meet our requirements.

## Pull Request Guide
Before you submit a pull request, check that it meets these guidelines:
1. Give your pull request a **helpful title** that summarizing your contribution.
2. Give your branch a **helpful name** summarizing your contribution (fork the repo and create a new branch for pull request).
3. Also, **add the issue number** which it addresses.
If there is no associated issue, feel free to [create one](https://github.com/jhfjhfj1/autokeras/issues).
4. Often pull requests resolve one or more other issues.
If merging your pull request means that some issues should be closed,
you should use keywords to link to them by following [this tutorial](https://blog.github.com/2013-05-14-closing-issues-via-pull-requests/).
5. For the case of bug fixes, at the time of the PR,
the test case should fail for the code base in the master branch and pass for the PR code.
6. Please prefix the title of your pull request with [MRG] if the contribution is complete and should be subjected to a detailed review.
 An incomplete contribution – where you expect to do more work before receiving a full review – should be prefixed [WIP] (to indicate a work in progress) and changed to [MRG] when it matures. 
7. Checkout from and pull request to the right branch. 
If it is a very urgent bug fix, checkout from master and pull request to both master and develop.
Otherwise, checkout from develop and pull request to develop.

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
Other than the base classes you have to extend,
there are some other classes you can extend.

### ModelTrainer
`autokeras.utils.ModelTrainer` is a class for training Pytorch models.
If needed a new metric or loss function, you can add your own to `loss_function.py` and `metric.py`.
You can follow its documentation to use it.
Make sure your loss function, metric, Pytorch model, and Dataloader are compatible with each other.
