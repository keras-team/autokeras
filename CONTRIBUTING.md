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
