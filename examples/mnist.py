import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
clf = ak.ImageClassifier(max_trials=3)
clf.fit(x_train, y_train, validation_data=(x_test, y_test))
y = clf.evaluate(x_test, y_test)
