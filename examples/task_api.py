import autokeras as ak
from keras.datasets import mnist,cifar10
import numpy as np

# Prepare the data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
if len(np.shape(x_train)) == 3:
    # If the raw image has 'Channel', we don't have to add one.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
# Search and train the classifier.
clf = ak.ImageClassifier(max_trials=100)
clf.fit(x_train, y_train, validation_data=(x_test, y_test))
y = clf.predict(x_test, y_test)
