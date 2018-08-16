from abc import ABC, abstractmethod


class Supervised(ABC):
    """The base classifier class.

    It is the base class for all the classifiers. It searches neural network architectures
    for the best configuration for the dataset.

    Attributes:
        verbose: A boolean value indicating the verbosity mode.
    """

    def __init__(self, verbose=False):
        """Initialize the instance.

        The classifier will be loaded from the files in 'path' if parameter 'resume' is True.
        Otherwise it would create a new one.

        Args:
            verbose: A boolean of whether the search process will be printed to stdout.

        """
        self.verbose = verbose

    @abstractmethod
    def fit(self, x_train=None, y_train=None, time_limit=None):
        """Find the best neural architecture and train it.

        Based on the given dataset, the function will find the best neural architecture for it.
        The dataset is in numpy.ndarray format.
        So they training data should be passed through `x_train`, `y_train`.

        Args:
            x_train: A numpy.ndarray instance containing the training data.
            y_train: A numpy.ndarray instance containing the label of the training data.
            time_limit: The time limit for the search in seconds.
        """

    @abstractmethod
    def predict(self, x_test):
        """Return predict results for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the results.
        """
        pass

    @abstractmethod
    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`."""
        pass
