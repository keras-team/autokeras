"""
Run the following commands first
pip3 install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc1
pip3 install autokeras==1.0.5

This Script searches for a model for the wine dataset
Source and Description of data:
"""
import os

import pandas as pd
import tensorflow as tf

import autokeras as ak

dataset_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
)

# save data
data_file_path = tf.keras.utils.get_file(
    fname=os.path.basename(dataset_url), origin=dataset_url
)

column_names = [
    "Wine",
    "Alcohol",
    "Malic.acid",
    "Ash",
    "Acl",
    "Mg",
    "Phenols",
    "Flavanoids",
    "Nonflavanoid.phenols",
    "Proanth",
    "Color.int",
    "Hue",
    "OD",
    "Proline",
]

feature_names = column_names[1:]
label_name = column_names[0]  # Wine

data = pd.read_csv(data_file_path, header=0, names=column_names)
# Shuffling
data = data.sample(frac=1)

split_length = int(data.shape[0] * 0.8)  # 141

# train and test
train_data = data.iloc[:split_length]
test_data = data.iloc[split_length:]


# Initialize the classifier.
clf = ak.StructuredDataClassifier(max_trials=5)

# Evaluate
clf.fit(x=train_data[feature_names], y=train_data[label_name])
print(
    "Accuracy: {accuracy}".format(
        accuracy=clf.evaluate(
            x=test_data[feature_names], y=test_data[label_name]
        )
    )
)
