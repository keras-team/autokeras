from abc import abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np
from autokeras.utils import validate_xy, resize_image_data, compute_image_resize_params
from autokeras.nn.metric import Accuracy
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.generator import ResNetGenerator, DenseNetGenerator
from autokeras.search import train
from autokeras.constant import Constant
from autokeras.preprocessor import ImageDataTransformer, OneHotEncoder
from autokeras.supervised import SingleModelSupervised


class PredefinedModel(SingleModelSupervised):
    """The base class for the predefined model without architecture search

    Attributes:
        graph: The graph form of the model.
        y_encoder: Label encoder, used in transform_y or inverse_transform_y for encode the label. For example,
                   if one hot encoder needed, y_encoder can be OneHotEncoder.
        data_transformer: A instance of transformer to process the data, See example as ImageDataTransformer.
        verbose: A boolean of whether the search process will be printed to stdout.
        path: A string. The path to a directory, where the intermediate results are saved.
    """

    def __init__(self, y_encoder=OneHotEncoder(), data_transformer=None, verbose=False, path=None):
        super().__init__(verbose, path)
        self.graph = None
        self.generator = None
        self.y_encoder = y_encoder
        self.data_transformer = data_transformer

    @abstractmethod
    def _init_generator(self, n_output_node, input_shape):
        """Initialize the generator to generate the model architecture.

        Args:
            n_output_node:  A integer value represent the number of output node in the final layer.
            input_shape: A tuple to express the shape of every train entry.
        """
        pass

    @property
    def loss(self):
        return classification_loss

    @property
    def metric(self):
        return Accuracy

    def preprocess(self, x):
        return resize_image_data(x, self.resize_shape)

    def transform_y(self, y_train):
        return self.y_encoder.transform(y_train)

    def inverse_transform_y(self, output):
        return self.y_encoder.inverse_transform(output)

    def fit(self, x, y, trainer_args=None):
        """Trains the model on the dataset given.

        Args:
            x: A numpy.ndarray instance containing the training data or the training data combined with the
               validation data.
            y: A numpy.ndarray instance containing the label of the training data. or the label of the training data
               combined with the validation label.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
        """
        validate_xy(x, y)
        self.resize_shape = compute_image_resize_params(x)
        x = self.preprocess(x)
        self.y_encoder.fit(y)
        y = self.transform_y(y)
        # Divide training data into training and testing data.
        validation_set_size = int(len(y) * Constant.VALIDATION_SET_SIZE)
        validation_set_size = min(validation_set_size, 500)
        validation_set_size = max(validation_set_size, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=validation_set_size,
                                                            random_state=42)

        # initialize data_transformer
        self.data_transformer = ImageDataTransformer(x_train)
        # Wrap the data into DataLoaders
        train_loader = self.data_transformer.transform_train(x_train, y_train)
        test_loader = self.data_transformer.transform_test(x_test, y_test)

        self.generator = self._init_generator(self.y_encoder.n_classes, x_train.shape[1:])
        graph = self.generator.generate()

        if trainer_args is None:
            trainer_args = {'max_no_improvement_num': 30}
        _, _1, self.graph = train(None, graph, train_loader, test_loader,
                                  trainer_args, self.metric, self.loss,
                                  self.verbose, self.path)


class PredefinedResnet(PredefinedModel):
    def _init_generator(self, n_output_node, input_shape):
        return ResNetGenerator(n_output_node, input_shape)


class PredefinedDensenet(PredefinedModel):
    def _init_generator(self, n_output_node, input_shape):
        return DenseNetGenerator(n_output_node, input_shape)
