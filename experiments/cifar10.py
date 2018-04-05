import sys
from keras.datasets import cifar10
import numpy as np

import os
import GPUtil

from autokeras.classifier import ImageClassifier


def select_gpu():
    try:
        # Get the first available GPU
        DEVICE_ID_LIST = GPUtil.getFirstAvailable()
        DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list

        # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    except FileNotFoundError:
        print("GPU not found")


if __name__ == '__main__':
    select_gpu()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))
    clf = ImageClassifier(searcher_type=sys.argv[1], path=sys.argv[2], verbose=False)

    clf.fit(x_train, y_train)
    clf.final_fit(x_train, y_train)
    y = clf.evaluate(x_test, y_test)
    print(y)
    # MLP for Pima Indians Dataset with 10-fold cross validation
    # scores = clf.cross_validate(X, Y, 2)
    # print(np.mean(scores))
    # print(np.std(scores))

    # split into input (X) and output (Y) variables
    # define 10-fold cross validation test harness
