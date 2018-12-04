import torch

from functools import reduce

import os
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np

from autokeras.constant import Constant
from autokeras.net_module import CnnModule
from autokeras.utils import rand_temp_folder_generator, pickle_from_file, validate_xy, pickle_to_file


class Pretrained(ABC):
    """The base class for all pretrained task.

    Attributes:
        verbose: A boolean value indicating the verbosity mode.
    """

    def __init__(self, cuda=False):
        """Initialize the instance.

        Args:
            cuda: A boolean of whether to use gpu or not
        """
        self.cuda = cuda
        self.model = None

    @abstractmethod
    def load(self, model_path=None, trained_device='gpu'):
        """load pretrained model into self.model

        if model_path is None, a .pth model file will be downloaded


        Args:
            model_path: path to the .pth file to be loaded. if is None, auto-download will be triggered.
            trained_device: 'gpu' or 'cpu', on gpu or cpu the pretrained model was trained
        """
        pass

    @abstractmethod
    def predict(self, img_path, output_file_path=None):
        """Return predict results for the given image

        Args:
            img_path: path to the image to be predicted
            output_file_path: if None: will only numerical results; else will also save the output image to the given path

        Returns:
            prediction results.
        """
        pass
