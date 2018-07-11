
from keras.datasets import mnist
import numpy as np

from keras.optimizers import Adam

from autokeras.classifier import ImageClassifier


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))
    trainer_args = {'max_iter_num': 0,
                    'batch_size': 128,
                    'optimizer': Adam,
                    'augment': False}
    clf = ImageClassifier(path='/home/haifeng/mnist', verbose=True, searcher_args={'trainer_args': trainer_args})

    clf.fit(x_train, y_train, time_limit=3*60)
    clf.load_searcher().export_json('/home/haifeng/mnist/test.json')
