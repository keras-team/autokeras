from keras.datasets import mnist

from autokeras.classifier import ImageClassifier

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    clf = ImageClassifier()
    clf.fit(x_train, y_train)
    y = clf.predict(x_test)
