import sys
from keras.datasets import mnist
import numpy as np

from autokeras import constant
from autokeras.classifier import ImageClassifier


if __name__ == '__main__':
    constant.MAX_MODEL_NUM = 1
    constant.MAX_ITER_NUM = 1
    constant.EPOCHS_EACH = 1
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))
    clf = ImageClassifier(searcher_type=sys.argv[1])
    clf.fit(x_train, y_train)
    y = clf.evaluate(x_test, y_test)
    # MLP for Pima Indians Dataset with 10-fold cross validation
    mean_score, std_dev = clf.cross_validate(X, Y, 2)

    # split into input (X) and output (Y) variables
    # define 10-fold cross validation test harness
