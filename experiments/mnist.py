import sys

from keras import backend
from keras.datasets import cifar10, fashion_mnist, mnist
import numpy as np

import os
import GPUtil
from keras.optimizers import Adadelta, Adam

from autokeras import constant
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from autokeras.classifier import ImageClassifier
from autokeras.graph import Graph
from autokeras.utils import ModelTrainer


def select_gpu():
    try:
        # Get the first available GPU
        DEVICE_ID_LIST = GPUtil.getFirstAvailable()
        DEVICE_ID = DEVICE_ID_LIST[0]  # grab first element from list

        # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    except FileNotFoundError:
        print("GPU not found")


if __name__ == '__main__':
    select_gpu()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
    # constant.LIMIT_MEMORY = True
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))
    trainer_args = {'max_iter_num': 12,
                    'batch_size': 128,
                    'optimizer': Adam,
                    'augment': False}
    clf = ImageClassifier(searcher_type='bayesian', path='/home/haifeng/mnist', verbose=True, trainer_args=trainer_args)

    clf.fit(x_train, y_train, time_limit=12*60*60)
    clf.final_fit(x_train, y_train, x_test, y_test, trainer_args, retrain=True)
    y = clf.evaluate(x_test, y_test)
    print(y)
    # MLP for Pima Indians Dataset with 10-fold cross validation
    # model = clf.load_searcher().load_best_model()
    clf.verbose = True
    # scores = clf.cross_validate(X, Y, 10)
    # print(scores)
    # print(np.mean(scores))
    # print(np.std(scores))

    # split into input (X) and output (Y) variables
    # define 10-fold cross validation test harness

    # k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=7)
    # y_train = clf.y_encoder.transform(y_train)
    # y_test = clf.y_encoder.transform(y_test)
    # ret = []
    # for train, test in k_fold.split(X, Y):
    #     graph = Graph(model, False)
    #     backend.clear_session()
    #     model = graph.produce_model()
    #     ModelTrainer(model, x_train, y_train, x_test, y_test, True).train_model(max_iter_num=12,
    #                                                                             batch_size=128, optimizer=Adam())
    #     scores = model.evaluate(x_test, y_test, verbose=True)
    #     ret.append(scores[1] * 100)
    # ret = np.array(ret)
    # print(ret)
    # print(np.mean(ret))
    # print(np.std(ret))
