## Contributing:

Contributions are welcome, and greatly appreciated! Every little bit helps, and credit will always be given.


There are many ways to contribute to Auto-Keras, with the most common ones being contribution of code or documentation to the project. If you find a typo in the documentation, or have made improvements, do not hesitate to submit a GitHub pull request ([Pull Request Guideline](https://github.com/jhfjhfj1/autokeras/blob/master/PULL_REQUEST_TEMPLATE.md)). 

### Types of Contributions:
#### Report Bugs:
Refer to our [Bug Reporting Guideline](https://github.com/jhfjhfj1/autokeras/blob/master/.github/ISSUE_TEMPLATE/bug_report.md) and report bugs at https://github.com/jhfjhfj1/autokeras/issues. 
#### Fix Bugs:
Look through the [GitHub issues](https://github.com/jhfjhfj1/autokeras/issues) for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.
#### Implement Features
Look through the [GitHub issues](https://github.com/jhfjhfj1/autokeras/issues) for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.
#### Write Documentation
Auto-Keras could always use more documentation, whether as part of the official [Auto-Keras docs](https://github.com/jhfjhfj1/autokeras/tree/master/mkdocs/docs), in docstrings, or even on the web in blog posts, articles, and such.
#### Submit Feedback
The best way to send feedback is to file an issue at  https://github.com/jhfjhfj1/autokeras/issues

#### Issue Tracker Tags
All issues and pull requests on the Github issue tracker should have (at least) one of the following tags:

  Tag | Description
  :--- | :---
  Bug:           |	Something is happening that clearly shouldn’t happen. Wrong results as well as unexpected errors go here.
  Enhancement:	 |  Improving performance, usability, consistency.                                                            
  Documentation: |  Missing, incorrect or sub-standard documentations and examples.                                            
  New Feature:	 |  Feature requests and pull requests implementing a new feature.                                             


## Code Style Guide:

This project tries to closely follow the official Python Style Guide detailed in [PEP8](https://www.python.org/dev/peps/pep-0008/).
The docstrings follow the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#381-docstrings).
Please follow these style guide closely, especially for the docstrings,
which would be extracted automatically to generate the documentation.

## API Guide:

You only need to read this guide if you are contributing or updating a new task module, e.g. TextClassifier, VideoClassifier.
In general, all new task module should inherit their objects from the `Classifier` class in `autokeras/classifier.py`.
Reach out to us if you feel there is a special requirement.
For every new feature, a new directory should be created inside the /autokeras directory, e.g. text_classifier.
All the code contributed should be within the directory.
The details of the functions to inherit is in the documentation of [`autokeras/classifier.py`]()
 
## Documentation Guide:

The documentation should be provided in two ways, docstring, tutorial, and readme file.
We prefer the documentation to be as complete as possible.

### Docstring
All the methods and classes may directly called by the user need to be documented with docstrings.
The docstrings should contain all the fields required by the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#381-docstrings).

### Tutorial
You only need to add tutorials to your code if you are contributing or updating a new task module,
e.g. TextClassifier, VideoClassifier,
or a new function could be directly called by the user.
You can modify `mkdocs/docs/start.md` to add your tutorial.
The code example of your new task module should be added to `examples` directory.

### Readme File
You only need to add tutorials to your code if you are contributing or updating a new task module,
e.g. TextClassifier, VideoClassifier.
The readme file should be named as `README.md`.
It should be written in Markdown.
The content should contain your name, affiliation, and any reference to the method you use.

## Testing Guide
[Pytest](https://docs.pytest.org/en/latest/) is used to write the unit tests of Auto-Keras.
You should test your code by writing unit testing code in `tests` directory.
The testing file name should be the `.py` file with a prefix of `test_` in corresponding directory,
e.g., the name should be `test_layers.py` if the code of which is to test `layer.py`.
The tests should be run in the root directory of the project by executing the `cov.sh` file.
It would output the coverage information into a directory named `htmlcov`.
Please make sure the code coverage percentage does not decrease after your contribution,
otherwise the code will not be merged.
