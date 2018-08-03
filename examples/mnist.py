from keras.datasets import mnist
from autokeras.classifier import ImageClassifier
from autokeras.constant import Constant

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    his = x_test.copy()

    Constant.SEARCH_MAX_ITER = 1
    Constant.MAX_ITER_NUM = 1
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=30)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    import numpy as np
    assert False not in np.equal(his, x_test)
    print(y)
