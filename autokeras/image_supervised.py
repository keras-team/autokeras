import csv
import os
import pickle
from abc import abstractmethod
from functools import reduce

import numpy as np
import torch
from scipy import ndimage
from sklearn.model_selection import train_test_split

from autokeras.cnn_module import CnnModule
from autokeras.constant import Constant
from autokeras.loss_function import classification_loss, regression_loss
from autokeras.metric import Accuracy, MSE
from autokeras.preprocessor import OneHotEncoder, DataTransformer
from autokeras.supervised import Supervised, PortableClass
from autokeras.utils import has_file, pickle_from_file, pickle_to_file, temp_folder_generator, validate_xy


def read_csv_file(csv_file_path):
    """Read the csv file and returns two separate list containing files name and their labels.

    Args:
        csv_file_path: Path to the CSV file.

    Returns:
        file_names: List containing files names.
        file_label: List containing their respective labels.
    """
    file_names = []
    file_labels = []
    with open(csv_file_path, 'r') as files_path:
        path_list = csv.DictReader(files_path)
        fieldnames = path_list.fieldnames
        for path in path_list:
            file_names.append(path[fieldnames[0]])
            file_labels.append(path[fieldnames[1]])
    return file_names, file_labels


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
                img = ndimage.imread(fname=img_path)
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


class ImageSupervised(Supervised):
    """The image classifier class.

    It is used for image classification. It searches convolutional neural network architectures
    for the best configuration for the dataset.

    Attributes:
        path: A path to the directory to save the classifier.
        y_encoder: An instance of OneHotEncoder for `y_train` (array of categorical labels).
        verbose: A boolean value indicating the verbosity mode.
        searcher: An instance of BayesianSearcher. It searches different
            neural architecture to find the best model.
        searcher_args: A dictionary containing the parameters for the searcher's __init__ function.
        augment: A boolean value indicating whether the data needs augmentation.  If not define, then it
                will use the value of Constant.DATA_AUGMENTATION which is True by default.
    """

    def __init__(self, verbose=False, path=None, resume=False, searcher_args=None, augment=None):
        """Initialize the instance.

        The classifier will be loaded from the files in 'path' if parameter 'resume' is True.
        Otherwise it would create a new one.

        Args:
            verbose: A boolean of whether the search process will be printed to stdout.
            path: A string. The path to a directory, where the intermediate results are saved.
            resume: A boolean. If True, the classifier will continue to previous work saved in path.
                Otherwise, the classifier will start a new search.
            augment: A boolean value indicating whether the data needs augmentation. If not define, then it
                will use the value of Constant.DATA_AUGMENTATION which is True by default.

        """
        super().__init__(verbose)

        if searcher_args is None:
            searcher_args = {}

        if path is None:
            path = temp_folder_generator()

        self.cnn = CnnModule(self.loss, self.metric, searcher_args, path, verbose)

        if augment is None:
            augment = Constant.DATA_AUGMENTATION

        self.path = path
        if has_file(os.path.join(self.path, 'classifier')) and resume:
            classifier = pickle_from_file(os.path.join(self.path, 'classifier'))
            self.__dict__ = classifier.__dict__
        else:
            self.y_encoder = None
            self.data_transformer = None
            self.verbose = verbose
            self.augment = augment

    @property
    @abstractmethod
    def metric(self):
        pass

    @property
    @abstractmethod
    def loss(self):
        pass

    def fit(self, x, y, x_test=None, y_test=None, time_limit=None):
        x = np.array(x)
        y = np.array(y).flatten()
        validate_xy(x, y)
        y = self.transform_y(y)
        if x_test is None or y_test is None:
            # Divide training data into training and testing data.
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=min(Constant.VALIDATION_SET_SIZE,
                                                                              int(len(y) * 0.2)),
                                                                random_state=42)
        else:
            x_train = x
            y_train = y
        # Transform x_train
        if self.data_transformer is None:
            self.data_transformer = DataTransformer(x, augment=self.augment)

        # Wrap the data into DataLoaders
        train_data = self.data_transformer.transform_train(x_train, y_train)
        test_data = self.data_transformer.transform_test(x_test, y_test)

        # Save the classifier
        pickle.dump(self, open(os.path.join(self.path, 'classifier'), 'wb'))
        pickle_to_file(self, os.path.join(self.path, 'classifier'))

        if time_limit is None:
            time_limit = 24 * 60 * 60

        self.cnn.fit(self.get_n_output_node(), x_train.shape, train_data, test_data, time_limit)

    @abstractmethod
    def get_n_output_node(self):
        pass

    def transform_y(self, y_train):
        return y_train

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
        model = self.cnn.best_model.produce_model()
        model.eval()

        outputs = []
        with torch.no_grad():
            for index, inputs in enumerate(test_loader):
                outputs.append(model(inputs).numpy())
        output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
        return self.inverse_transform_y(output)

    def inverse_transform_y(self, output):
        return output

    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`."""
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_test, y_predict)

    def final_fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=False):
        """Final training after found the best architecture.

        Args:
            x_train: A numpy.ndarray of training data.
            y_train: A numpy.ndarray of training targets.
            x_test: A numpy.ndarray of testing data.
            y_test: A numpy.ndarray of testing targets.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
            retrain: A boolean of whether reinitialize the weights of the model.
        """
        if trainer_args is None:
            trainer_args = {'max_no_improvement_num': 30}

        y_train = self.transform_y(y_train)
        y_test = self.transform_y(y_test)

        train_data = self.data_transformer.transform_train(x_train, y_train)
        test_data = self.data_transformer.transform_test(x_test, y_test)

        self.cnn.final_fit(train_data, test_data, trainer_args, retrain)

    def load_searcher(self):
        return pickle_from_file(os.path.join(self.path, 'searcher'))

    def export_keras_model(self, model_file_name):
        """ Exports the best Keras model to the given filename. """
        self.cnn.best_model.produce_keras_model().save(model_file_name)

    def export_autokeras_model(self, model_file_name):
        """ Creates and Exports the AutoKeras model to the given filename. """
        portable_model = PortableImageSupervised(graph=self.cnn.best_model,
                                                 y_encoder=self.y_encoder,
                                                 data_transformer=self.data_transformer,
                                                 metric=self.metric,
                                                 inverse_transform_y_method=self.inverse_transform_y)
        pickle_to_file(portable_model, model_file_name)


class ImageClassifier(ImageSupervised):
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


class ImageRegressor(ImageSupervised):
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


class PortableImageSupervised(PortableClass):
    def __init__(self, graph, data_transformer, y_encoder, metric, inverse_transform_y_method):
        """Initialize the instance.
        Args:
            graph: The graph form of the learned model
        """
        super().__init__(graph)
        self.data_transformer = data_transformer
        self.y_encoder = y_encoder
        self.metric = metric
        self.inverse_transform_y_method = inverse_transform_y_method

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
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_test, y_predict)
