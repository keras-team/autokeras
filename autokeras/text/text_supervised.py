from abc import ABC

import numpy as np

from autokeras.nn.loss_function import classification_loss, regression_loss
from autokeras.nn.metric import Accuracy, MSE
from autokeras.preprocessor import OneHotEncoder, TextDataTransformer
from autokeras.supervised import DeepSupervised
from autokeras.text.text_preprocessor import text_preprocess


class TextSupervised(DeepSupervised, ABC):
    """TextClassifier class.

    Attributes:
        cnn: CNN module from net_module.py.
        path: A path to the directory to save the classifier as well as intermediate results.
        y_encoder: Label encoder, used in transform_y or inverse_transform_y for encode the label. For example,
                    if one hot encoder needed, y_encoder can be OneHotEncoder.
        data_transformer: A transformer class to process the data. See example as ImageDataTransformer.
        verbose: A boolean value indicating the verbosity mode which determines whether the search process
                will be printed to stdout.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        The classifier will be loaded from the files in 'path' if parameter 'resume' is True.
        Otherwise it would create a new one.
        Args:
            verbose: A boolean of whether the search process will be printed to stdout.
            path: A string. The path to a directory, where the intermediate results are saved.
            resume: A boolean. If True, the classifier will continue to previous work saved in path.
                Otherwise, the classifier will start a new search.
            searcher_args: A dictionary containing the parameters for the searcher's __init__ function.
        """
        super().__init__(**kwargs)

    def fit(self, x, y, x_test=None, y_test=None, batch_size=None, time_limit=None):
        """Find the best neural architecture and train it.

        Based on the given dataset, the function will find the best neural architecture for it.
        The dataset is in numpy.ndarray format.
        So they training data should be passed through `x_train`, `y_train`.

        Args:
            x: A numpy.ndarray instance containing the training data.
            y: A numpy.ndarray instance containing the label of the training data.
            y_test: A numpy.ndarray instance containing the testing data.
            x_test: A numpy.ndarray instance containing the label of the testing data.
            batch_size: int, define the batch size.
            time_limit: The time limit for the search in seconds.
        """
        x = text_preprocess(x)

        x = np.array(x)
        y = np.array(y).flatten()

        super().fit(x, y, x_test, y_test, time_limit)

    def init_transformer(self, x):
        # Wrap the data into DataLoaders
        if self.data_transformer is None:
            self.data_transformer = TextDataTransformer()

    def preprocess(self, x):
        return text_preprocess(x)


class TextClassifier(TextSupervised):
    @property
    def metric(self):
        return Accuracy

    @property
    def loss(self):
        return classification_loss

    def transform_y(self, y_train):
        # Transform y_train.
        if self.y_encoder is None:
            self.y_encoder = OneHotEncoder()
            self.y_encoder.fit(y_train)
        y_train = self.y_encoder.transform(y_train)
        return y_train

    def inverse_transform_y(self, output):
        return self.y_encoder.inverse_transform(output)

    def get_n_output_node(self):
        return self.y_encoder.n_classes


class TextRegressor(TextClassifier):
    """ TextRegressor class.

    It is used for text regression. It searches convolutional neural network architectures
    for the best configuration for the text dataset.
    """

    @property
    def loss(self):
        return regression_loss

    @property
    def metric(self):
        return MSE

    def get_n_output_node(self):
        return 1

    def transform_y(self, y_train):
        return y_train.flatten().reshape(len(y_train), 1)

    def inverse_transform_y(self, output):
        return output.flatten()
