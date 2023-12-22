"""
Search for a good model for the
[MNIST](https://keras.io/datasets/#mnist-database-of-handwritten-digits)
dataset.
"""

from keras.datasets import mnist

import autokeras as ak

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)

# Initialize the ImageClassifier.
clf = ak.ImageClassifier(max_trials=3)
# Search for the best model.
clf.fit(x_train, y_train, epochs=10)
# Evaluate on the testing data.
print("Accuracy: {accuracy}".format(accuracy=clf.evaluate(x_test, y_test)))
