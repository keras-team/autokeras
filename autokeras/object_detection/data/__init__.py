from .voc0712 import VOC_CLASSES
from .config import *
import torch
import cv2
import numpy as np


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

