from abc import ABC

from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.supervised import SingleModelSupervised


class TextClassifier(SingleModelSupervised, ABC):
    """TextClassifier class.
    """

    def __init__(self, **kwargs):
        super().__init_(**kwargs)

    def fit(self, x, y, time_limit=None):
        pass

    @property
    def metric(self):
        return Accuracy

    @property
    def loss(self):
        return classification_loss

    def preprocess(self, x):
        pass

    def transform_y(self, y_train):
        pass

    def inverse_transform_y(self, output):
        pass
