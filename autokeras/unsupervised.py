from abc import ABC, abstractmethod


class Unsupervised(ABC):
    """ The base class for all unsupervised task

    Attributes:
        verbose: A boolean value indicating the verbosity mode.
    """

    def __init__(self, verbose=False):
        """
            Args:
            verbose: A boolean of whether the search process will be printed to stdout.
        """
        self.verbose = verbose

    @abstractmethod
    def fit(self, x_train):
        """

        Args:
            x_train: A numpy.ndarray instance containing the training data.
        """
        pass

    @abstractmethod
    def generate(self, input_sample=None):
        """
        Args: A numpy.ndarray or torch.tensor input fed into the model to generate the output

        Returns: the result of applying the model
        """
        pass

