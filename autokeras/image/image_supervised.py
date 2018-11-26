import os
from abc import ABC
from functools import reduce

import numpy as np
import torch

from autokeras.constant import Constant
from autokeras.nn.loss_function import classification_loss, regression_loss
from autokeras.nn.metric import Accuracy, MSE
from autokeras.preprocessor import OneHotEncoder, ImageDataTransformer
from autokeras.supervised import PortableClass, DeepSupervised
from autokeras.utils import pickle_to_file, \
    read_csv_file, read_image, compute_image_resize_params, resize_image_data


def read_images(img_file_names, images_dir_path):
    """Read the images from the path and return their numpy.ndarray instance.
        Return a numpy.ndarray instance containing the training data.

    Args:
        img_file_names: List containing images names.
        images_dir_path: Path to the directory containing images.
    """
    x_train = []
    if os.path.isdir(images_dir_path):
        for img_file in img_file_names:
            img_path = os.path.join(images_dir_path, img_file)
            if os.path.exists(img_path):
                img = read_image(img_path)
                if len(img.shape) < 3:
                    img = img[..., np.newaxis]
                x_train.append(img)
            else:
                raise ValueError("%s image does not exist" % img_file)
    else:
        raise ValueError("Directory containing images does not exist")
    return np.asanyarray(x_train)


def load_image_dataset(csv_file_path, images_path):
    """Load images from the files and labels from a csv file.

    Second, the dataset is a set of images and the labels are in a CSV file.
    The CSV file should contain two columns whose names are 'File Name' and 'Label'.
    The file names in the first column should match the file names of the images with extensions,
    e.g., .jpg, .png.
    The path to the CSV file should be passed through the `csv_file_path`.
    The path to the directory containing all the images should be passed through `image_path`.

    Args:
        csv_file_path: CSV file path.
        images_path: Path where images exist.

    Returns:
        x: Four dimensional numpy.ndarray. The channel dimension is the last dimension.
        y: The labels.
    """
    img_file_name, y = read_csv_file(csv_file_path)
    x = read_images(img_file_name, images_path)
    return np.array(x), np.array(y)


class ImageSupervised(DeepSupervised, ABC):
    """Abstract image supervised class.

    Attributes:
        path: A path to the directory to save the classifier as well as intermediate results.
        cnn: CNN module from net_module.py.
        y_encoder: Label encoder, used in transform_y or inverse_transform_y for encode the label. For example,
                    if one hot encoder needed, y_encoder can be OneHotEncoder.
        data_transformer: A transformer class to process the data. See example as ImageDataTransformer.
        verbose: A boolean value indicating the verbosity mode which determines whether the search process
                will be printed to stdout.
        augment: A boolean value indicating whether the data needs augmentation.  If not define, then it
                will use the value of Constant.DATA_AUGMENTATION which is True by default.
        searcher_args: A dictionary containing the parameters for the searcher's __init__ function.
        resize_height: resize image height.
        resize_width: resize image width.
    """

    def __init__(self, augment=None, **kwargs):
        """Initialize the instance.
        The classifier will be loaded from the files in 'path' if parameter 'resume' is True.
        Otherwise it would create a new one.
        Args:
            verbose: A boolean of whether the search process will be printed to stdout.
            path: A string. The path to a directory, where the intermediate results are saved.
            resume: A boolean. If True, the classifier will continue to previous work saved in path.
                Otherwise, the classifier will start a new search.
            searcher_args: A dictionary containing the parameters for the searcher's __init__ function.
            augment: A boolean value indicating whether the data needs augmentation. If not define, then it
                will use the value of Constant.DATA_AUGMENTATION which is True by default.
        """
        if augment is None:
            augment = Constant.DATA_AUGMENTATION
        self.augment = augment
        self.resize_height = None
        self.resize_width = None

        super().__init__(**kwargs)

    def fit(self, x, y, x_test=None, y_test=None, time_limit=None):
        x = np.array(x)
        y = np.array(y).flatten()

        if self.verbose:
            print("Preprocessing the images.")

        if x is not None and (len(x.shape) == 4 or len(x.shape) == 1 and len(x[0].shape) == 3):
            self.resize_height, self.resize_width = compute_image_resize_params(x)

        if self.resize_height is not None:
            x = resize_image_data(x, self.resize_height, self.resize_width)
            print("x is ", x.shape)

        if self.resize_height is not None:
            x_test = resize_image_data(x_test, self.resize_height, self.resize_width)

        if self.verbose:
            print("Preprocessing finished.")

        super().fit(x, y, x_test, y_test, time_limit)

    def init_transformer(self, x):
        if self.data_transformer is None:
            self.data_transformer = ImageDataTransformer(x, augment=self.augment)

    def export_autokeras_model(self, model_file_name):
        """ Creates and Exports the AutoKeras model to the given filename. """
        portable_model = PortableImageSupervised(graph=self.cnn.best_model,
                                                 y_encoder=self.y_encoder,
                                                 data_transformer=self.data_transformer,
                                                 metric=self.metric,
                                                 inverse_transform_y_method=self.inverse_transform_y,
                                                 resize_params=(self.resize_height, self.resize_width))
        pickle_to_file(portable_model, model_file_name)

    def preprocess(self, x):
        if len(x.shape) != 0 and len(x[0].shape) == 3:
            if self.resize_height is not None:
                return resize_image_data(x, self.resize_height, self.resize_width)
        return x


class ImageClassifier(ImageSupervised):
    """ImageClassifier class.

    It is used for image classification. It searches convolutional neural network architectures
    for the best configuration for the image dataset.
    """

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

    @property
    def metric(self):
        return Accuracy


class ImageClassifier1D(ImageClassifier):
    """ ImageClassifier1D class.

    It is used for 1D image classification. It searches convolutional neural network architectures
    for the best configuration for the 1D image dataset.
    """

    def __init__(self, **kwargs):
        kwargs['augment'] = False
        super().__init__(**kwargs)


class ImageClassifier3D(ImageClassifier):
    """ ImageClassifier3D class.

    It is used for 3D image classification. It searches convolutional neural network architectures
    for the best configuration for the 1D image dataset.
    """

    def __init__(self, **kwargs):
        kwargs['augment'] = False
        super().__init__(**kwargs)


class ImageRegressor(ImageSupervised):
    """ImageRegressor class.

    It is used for image regression. It searches convolutional neural network architectures
    for the best configuration for the image dataset.
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


class ImageRegressor1D(ImageRegressor):
    """ ImageRegressor1D class.

    It is used for 1D image regression. It searches convolutional neural network architectures
    for the best configuration for the 1D image dataset.
    """

    def __init__(self, **kwargs):
        kwargs['augment'] = False
        super().__init__(**kwargs)


class ImageRegressor3D(ImageRegressor):
    """ ImageRegressor3D class.

    It is used for 3D image regression. It searches convolutional neural network architectures
    for the best configuration for the 1D image dataset.
    """

    def __init__(self, **kwargs):
        kwargs['augment'] = False
        super().__init__(**kwargs)


class PortableImageSupervised(PortableClass):
    def __init__(self, graph, data_transformer, y_encoder, metric, inverse_transform_y_method, resize_params):
        """Initialize the instance.
        Args:
            graph: The graph form of the learned model
        """
        super().__init__(graph)
        self.data_transformer = data_transformer
        self.y_encoder = y_encoder
        self.metric = metric
        self.inverse_transform_y_method = inverse_transform_y_method
        self.resize_height = resize_params[0]
        self.resize_width = resize_params[1]

    def predict(self, x_test):
        """Return predict results for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the results.
        """
        if Constant.LIMIT_MEMORY:
            pass

        test_loader = self.data_transformer.transform_test(x_test)
        model = self.graph.produce_model()
        model.eval()

        outputs = []
        with torch.no_grad():
            for index, inputs in enumerate(test_loader):
                outputs.append(model(inputs).numpy())
        output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
        return self.inverse_transform_y(output)

    def inverse_transform_y(self, output):
        return self.inverse_transform_y_method(output)

    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`."""
        if self.resize_height is not None:
            x_test = resize_image_data(x_test, self.resize_height, self.resize_width)
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_test, y_predict)
