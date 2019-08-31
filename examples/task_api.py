import autokeras as ak
from keras.datasets import mnist

# Prepare the data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))
# Search and train the classifier.
clf = ak.ImageClassifier(max_trials=100)
clf.fit(x_train, y_train, validation_data=(x_test, y_test))
y = clf.predict(x_test, y_test)
