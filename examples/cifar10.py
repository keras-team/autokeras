from tensorflow.keras.datasets import cifar10

import autokeras as ak

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Initialize the ImageClassifier.
clf = ak.ImageClassifier(max_trials=3)
# Search for the best model.
clf.fit(x_train, y_train, epochs=5)
# Evaluate on the testing data.
print("Accuracy: {accuracy}".format(accuracy=clf.evaluate(x_test, y_test)[1]))
