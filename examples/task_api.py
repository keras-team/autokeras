import autokeras as ak
from tensorflow.python.keras.datasets import mnist, cifar10
import numpy as np

# Prepare the data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
data_slice = 50
x_train = x_train[:data_slice]
y_train = y_train[:data_slice]
x_test = x_test[:data_slice]
y_test = y_test[:data_slice]
x_train = x_train.astype(np.float64)
x_test = x_test.astype(np.float64)
if len(np.shape(x_train)) == 3:
    # If the raw image has 'Channel', we don't have to add one.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
# Search and train the classifier.
clf = ak.ImageClassifier(max_trials=3)
clf.fit(x_train, y_train, validation_data=(x_test, y_test))
y = clf.predict(x_test, y_test)
