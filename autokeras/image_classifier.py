import os
import pickle
import csv
import time
from functools import reduce

import torch

import scipy.ndimage as ndimage

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from autokeras.constant import Constant
from autokeras.preprocessor import OneHotEncoder, DataTransformer
from autokeras.search import BayesianSearcher, train
from autokeras.utils import ensure_dir, has_file, pickle_from_file, pickle_to_file
from autokeras.classifier import read_csv_file, Classifier

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


class ImageClassifier(Classifier):
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
        augment: A boolean value indicating whether the data needs augmentation.
    """
    pass