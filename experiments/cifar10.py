import sys
from keras.datasets import cifar10
import numpy as np

import os
import GPUtil

from autokeras import constant
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
    # constant.LIMIT_MEMORY = True
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))
    clf = ImageClassifier(searcher_type='bayesian', path='/home/haifeng/cifar10', verbose=False)

    clf.fit(x_train, y_train, time_limit=12*60*60)
    # clf.final_fit(x_train, y_train)
    y = clf.evaluate(x_test, y_test)
    print(y)
    # MLP for Pima Indians Dataset with 10-fold cross validation
    scores = clf.cross_validate(X, Y, 10)
    print(scores)
    print(np.mean(scores))
    print(np.std(scores))

    # split into input (X) and output (Y) variables
    # define 10-fold cross validation test harness
