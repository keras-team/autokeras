"""shell
pip install autokeras
"""

import os

import keras
import numpy as np
from sklearn.datasets import load_files

import autokeras as ak

"""
## A Simple Example
The first step is to prepare your data. Here we use the [IMDB
dataset](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification)
as an example.
"""


dataset = keras.utils.get_file(
    fname="aclImdb.tar.gz",
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    extract=True,
)

# set path to dataset
IMDB_DATADIR = os.path.join(
    os.path.dirname(dataset), "aclImdb_extracted", "aclImdb"
)

classes = ["pos", "neg"]
train_data = load_files(
    os.path.join(IMDB_DATADIR, "train"), shuffle=True, categories=classes
)
test_data = load_files(
    os.path.join(IMDB_DATADIR, "test"), shuffle=False, categories=classes
)

x_train = np.array(train_data.data)[:100]
y_train = np.array(train_data.target)[:100]
x_test = np.array(test_data.data)[:100]
y_test = np.array(test_data.target)[:100]

print(x_train.shape)  # (25000,)
print(y_train.shape)  # (25000, 1)
print(x_train[0][:50])  # this film was just brilliant casting

"""
The second step is to run the [TextClassifier](/text_classifier).  As a quick
demo, we set epochs to 2.  You can also leave the epochs unspecified for an
adaptive number of epochs.
"""


# Initialize the text classifier.
clf = ak.TextClassifier(
    overwrite=True, max_trials=1
)  # It only tries 1 model as a quick demo.
# Feed the text classifier with training data.
clf.fit(x_train, y_train, epochs=1, batch_size=2)
# Predict with the best model.
predicted_y = clf.predict(x_test)
# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))


"""
## Validation Data
By default, AutoKeras use the last 20% of training data as validation data.  As
shown in the example below, you can use `validation_split` to specify the
percentage.
"""

clf.fit(
    x_train,
    y_train,
    # Split the training data and use the last 15% as validation data.
    validation_split=0.15,
    epochs=1,
    batch_size=2,
)

"""
You can also use your own validation set instead of splitting it from the
training data with `validation_data`.
"""

split = 5
x_val = x_train[split:]
y_val = y_train[split:]
x_train = x_train[:split]
y_train = y_train[:split]
clf.fit(
    x_train,
    y_train,
    epochs=1,
    # Use your own validation set.
    validation_data=(x_val, y_val),
    batch_size=2,
)

"""
## Customized Search Space
For advanced users, you may customize your search space by using
[AutoModel](/auto_model/#automodel-class) instead of
[TextClassifier](/text_classifier). You can configure the
[TextBlock](/block/#textblock-class) for some high-level configurations. You can
also do not specify these arguments, which would leave the different choices to
be tuned automatically.  See the following example for detail.
"""


input_node = ak.TextInput()
output_node = ak.TextBlock()(input_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=1
)
clf.fit(x_train, y_train, epochs=1, batch_size=2)

"""
## Reference
[TextClassifier](/text_classifier),
[AutoModel](/auto_model/#automodel-class),
[ConvBlock](/block/#convblock-class),
[TextInput](/node/#textinput-class),
[ClassificationHead](/block/#classificationhead-class).
"""
