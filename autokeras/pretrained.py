from abc import ABC, abstractmethod


class Pretrained(ABC):
    """The base class for all pretrained task.
    Attributes:
        verbose: A boolean value indicating the verbosity mode.
    """

    def __init__(self):
        """Initialize the instance.
        """
        self.model = None

    @abstractmethod
    def load(self):
        """load pretrained model into self.model
        """
        pass

    @abstractmethod
    def predict(self, x_predict):
        """Return predict results for the given image
        Args:
            x_predict: An instance of numpy.ndarray containing the testing data.
        Returns:
            A numpy.ndarray containing the results.
        """
        pass