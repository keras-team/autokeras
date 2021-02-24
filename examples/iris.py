"""shell
pip install -q -U autokeras==1.0.5
pip install -q git+https://github.com/keras-team/keras-tuner.git@1.0.2rc1
"""

import os

import pandas as pd
import tensorflow as tf

import autokeras as ak

"""
Search for a good model for the
[iris](https://www.tensorflow.org/datasets/catalog/iris) dataset.
"""


# Prepare the dataset.
train_dataset_url = (
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
)
train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url), origin=train_dataset_url
)

test_dataset_url = (
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
)
test_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(test_dataset_url), origin=test_dataset_url
)

column_names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]
feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ["Iris setosa", "Iris versicolor", "Iris virginica"]

train = pd.read_csv(train_dataset_fp, names=column_names, header=0)

test = pd.read_csv(test_dataset_fp, names=column_names, header=0)

print(train.shape)  # (120, 5)
print(test.shape)  # (30, 5)

# Initialize the StructuredDataClassifier.
clf = ak.StructuredDataClassifier(
    max_trials=5,
    overwrite=True,
)
# Search for the best model with EarlyStopping.
cbs = [
    tf.keras.callbacks.EarlyStopping(patience=3),
]

clf.fit(
    x=train[feature_names],
    y=train[label_name],
    epochs=200,
    callbacks=cbs,
)
# Evaluate on the testing data.
print(
    "Accuracy: {accuracy}".format(
        accuracy=clf.evaluate(x=test[feature_names], y=test[label_name])
    )
)
