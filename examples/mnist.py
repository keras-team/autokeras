from keras.datasets import mnist
from autokeras import ImageClassifier
import tensorflow

if __name__ == '__main__':
    print(tensorflow.__version__)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    print(y * 100)
