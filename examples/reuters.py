"""shell
!pip install -q -U pip
!pip install -q -U autokeras==1.0.8
!pip install -q git+https://github.com/keras-team/keras-tuner.git@1.0.2rc1
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters

import autokeras as ak

"""
Search for a good model for the
[Reuters](https://keras.io/ja/datasets/#_5) dataset.
"""


# Prepare the dataset.
def reuters_raw(max_features=20000):

    index_offset = 3  # word index offset

    (x_train, y_train), (x_test, y_test) = reuters.load_data(
        num_words=max_features, index_from=index_offset
    )
    x_train = x_train
    y_train = y_train.reshape(-1, 1)
    x_test = x_test
    y_test = y_test.reshape(-1, 1)

    word_to_id = reuters.get_word_index()
    word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}
    x_train = list(
        map(lambda sentence: " ".join(id_to_word[i] for i in sentence), x_train)
    )
    x_test = list(
        map(lambda sentence: " ".join(id_to_word[i] for i in sentence), x_test)
    )
    x_train = np.array(x_train, dtype=np.str)
    x_test = np.array(x_test, dtype=np.str)
    return (x_train, y_train), (x_test, y_test)


# Prepare the data.
(x_train, y_train), (x_test, y_test) = reuters_raw()
print(x_train.shape)  # (8982,)
print(y_train.shape)  # (8982, 1)
print(x_train[0][:50])  # <START> <UNK> <UNK> said as a result of its decemb

# Initialize the TextClassifier
clf = ak.TextClassifier(
    max_trials=5,
    overwrite=True,
)

# Callback to avoid overfitting with the EarlyStopping.
cbs = [
    tf.keras.callbacks.EarlyStopping(patience=3),
]

# Search for the best model.
clf.fit(x_train, y_train, epochs=10, callback=cbs)

# Evaluate on the testing data.
print("Accuracy: {accuracy}".format(accuracy=clf.evaluate(x_test, y_test)))
