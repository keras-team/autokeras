from abc import ABC, abstractmethod


class Pretrained(ABC):
    """The base class for all pretrained task."""

    def __init__(self):
        """Initialize the instance."""
        self.model = None

    @abstractmethod
    def load(self, **kwargs):
        """Load pretrained model into self.model."""
        pass

    @abstractmethod
    def predict(self, input_data, **kwargs):
        """Return predict results for the given image
        Returns:
            A numpy.ndarray containing the results.
        """
        pass


