import os
from abc import ABC
import numpy as np
from multiprocessing import Pool, cpu_count

from autokeras.constant import Constant
from autokeras.nn.loss_function import classification_loss, regression_loss
from autokeras.nn.metric import Accuracy, MSE
from autokeras.preprocessor import OneHotEncoder, ImageDataTransformer
from autokeras.supervised import PortableDeepSupervised, DeepTaskSupervised
from autokeras.utils import pickle_to_file, \
    read_csv_file, read_image, compute_image_resize_params, resize_image_data


def _image_to_array(img_path):
    """Read the image from the path and return image object.
        Return an image object.

    Args:
        img_path: image file name in images_dir_path.
    """
    if os.path.exists(img_path):
        img = read_image(img_path)
        if len(img.shape) < 3:
            img = img[..., np.newaxis]
        return img
    else:
        raise ValueError("%s image does not exist" % img_path)


def read_images(img_file_names, images_dir_path, parallel=True):
    """Read the images from the path and return their numpy.ndarray instance.
        Return a numpy.ndarray instance containing the training data.

    Args:
        img_file_names: List containing images names.
        images_dir_path: Path to the directory containing images.
        parallel: (Default: True) Run _image_to_array will use multiprocessing.
    """
    img_paths = [os.path.join(images_dir_path, img_file)
                 for img_file in img_file_names]

    if os.path.isdir(images_dir_path):
        if parallel:
            pool = Pool(processes=cpu_count())
            x_train = pool.map(_image_to_array, img_paths)
            pool.close()
            pool.join()
        else:
            x_train = [_image_to_array(img_path) for img_path in img_paths]
    else:
        raise ValueError("Directory containing images does not exist")
    return np.asanyarray(x_train)


def load_image_dataset(csv_file_path, images_path, parallel=True):
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
        parallel: (Default: True) Load dataset with multiprocessing.

    Returns:
        x: Four dimensional numpy.ndarray. The channel dimension is the last dimension.
        y: The labels.
    """
    img_file_name, y = read_csv_file(csv_file_path)
    x = read_images(img_file_name, images_path, parallel)
    return np.array(x), np.array(y)


class ImageSupervised(DeepTaskSupervised, ABC):
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
        self.augment = augment if augment is not None else Constant.DATA_AUGMENTATION
        self.resize_shape = []

        super().__init__(**kwargs)

    def fit(self, x, y, time_limit=None):
        x = np.array(x)
        y = np.array(y)

        if self.verbose:
            print("Preprocessing the images.")

        self.resize_shape = compute_image_resize_params(x)

        x = resize_image_data(x, self.resize_shape)

        if self.verbose:
            print("Preprocessing finished.")

        super().fit(x, y, time_limit)

    def init_transformer(self, x):
        if self.data_transformer is None:
            self.data_transformer = ImageDataTransformer(
                x, augment=self.augment)

    def preprocess(self, x):
        return resize_image_data(x, self.resize_shape)


class ImageClassifier(ImageSupervised):
    """ImageClassifier class.

    It is used for image classification. It searches convolutional neural network architectures
    for the best configuration for the image dataset.
    """

    @property
    def loss(self):
        return classification_loss

    @property
    def metric(self):
        return Accuracy

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

    def export_autokeras_model(self, model_file_name):
        """ Creates and Exports the AutoKeras model to the given filename. """
        portable_model = PortableImageClassifier(graph=self.cnn.best_model,
                                                 y_encoder=self.y_encoder,
                                                 data_transformer=self.data_transformer,
                                                 resize_params=self.resize_shape,
                                                 path=self.path)
        pickle_to_file(portable_model, model_file_name)


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

    def export_autokeras_model(self, model_file_name):
        """ Creates and Exports the AutoKeras model to the given filename. """
        portable_model = PortableImageRegressor(graph=self.cnn.best_model,
                                                y_encoder=self.y_encoder,
                                                data_transformer=self.data_transformer,
                                                resize_params=self.resize_shape,
                                                path=self.path)
        pickle_to_file(portable_model, model_file_name)


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


class PortableImageSupervised(PortableDeepSupervised, ABC):
    def __init__(self, graph, y_encoder, data_transformer, resize_params, verbose=False, path=None):
        """Initialize the instance.
        Args:
            graph: The graph form of the learned model
        """
        super().__init__(graph, y_encoder, data_transformer, verbose, path)
        self.resize_shape = resize_params

    def preprocess(self, x):
        return resize_image_data(x, self.resize_shape)


class PortableImageClassifier(PortableImageSupervised):
    @property
    def loss(self):
        return classification_loss

    @property
    def metric(self):
        return Accuracy

    def transform_y(self, y_train):
        return self.y_encoder.transform(y_train)

    def inverse_transform_y(self, output):
        return self.y_encoder.inverse_transform(output)


class PortableImageRegressor(PortableImageSupervised):
    @property
    def loss(self):
        return regression_loss

    @property
    def metric(self):
        return MSE

    def transform_y(self, y_train):
        return y_train.flatten().reshape(len(y_train), 1)

    def inverse_transform_y(self, output):
        return output.flatten()
