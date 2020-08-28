# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os

import numpy as np
import tensorflow as tf

import autokeras as ak

SEED = 5
COLUMN_NAMES = [
    "sex",
    "age",
    "n_siblings_spouses",
    "parch",
    "fare",
    "class",
    "deck",
    "embark_town",
    "alone",
]
COLUMN_TYPES = {
    "sex": "categorical",
    "age": "numerical",
    "n_siblings_spouses": "categorical",
    "parch": "categorical",
    "fare": "numerical",
    "class": "categorical",
    "deck": "categorical",
    "embark_town": "categorical",
    "alone": "categorical",
}
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

TRAIN_CSV_PATH = tf.keras.utils.get_file(
    fname=os.path.basename(TRAIN_DATA_URL),
    origin=TRAIN_DATA_URL
)
TEST_CSV_PATH = tf.keras.utils.get_file(
    fname=os.path.basename(TEST_DATA_URL),
    origin=TEST_DATA_URL
)


def generate_data(num_instances=100, shape=(32, 32, 3), dtype="np"):
    np.random.seed(SEED)
    data = np.random.rand(*((num_instances,) + shape))
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    if dtype == "np":
        return data
    if dtype == "dataset":
        return tf.data.Dataset.from_tensor_slices(data)


def generate_one_hot_labels(num_instances=100, num_classes=10, dtype="np"):
    np.random.seed(SEED)
    labels = np.random.randint(num_classes, size=num_instances)
    data = tf.keras.utils.to_categorical(labels)
    if dtype == "np":
        return data
    if dtype == "dataset":
        return tf.data.Dataset.from_tensor_slices(data).batch(32)


def imdb_raw(num_instances=100):
    index_offset = 3  # word index offset

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=1000, index_from=index_offset
    )
    if num_instances is not None:
        x_train = x_train[:num_instances]
        y_train = y_train[:num_instances].reshape(-1, 1)
        x_test = x_test[:num_instances]
        y_test = y_test[:num_instances].reshape(-1, 1)

    word_to_id = tf.keras.datasets.imdb.get_word_index()
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


def build_graph():
    tf.keras.backend.clear_session()
    image_input = ak.ImageInput(shape=(32, 32, 3))
    merged_outputs = ak.ImageBlock()(image_input)
    head = ak.ClassificationHead(num_classes=10)
    head.output_shape = (10,)
    classification_outputs = head(merged_outputs)
    return ak.graph.Graph(inputs=image_input, outputs=classification_outputs)


def get_func_args(func):
    params = inspect.signature(func).parameters.keys()
    return set(params) - set(["self", "args", "kwargs"])
