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
from sklearn.datasets import load_files

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
    fname=os.path.basename(TRAIN_DATA_URL), origin=TRAIN_DATA_URL
)
TEST_CSV_PATH = tf.keras.utils.get_file(
    fname=os.path.basename(TEST_DATA_URL), origin=TEST_DATA_URL
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
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True,
    )

    # set path to dataset
    IMDB_DATADIR = os.path.join(os.path.dirname(dataset), "aclImdb")

    classes = ["pos", "neg"]
    train_data = load_files(
        os.path.join(IMDB_DATADIR, "train"), shuffle=True, categories=classes
    )
    test_data = load_files(
        os.path.join(IMDB_DATADIR, "test"), shuffle=False, categories=classes
    )

    x_train = np.array(train_data.data)
    y_train = np.array(train_data.target)
    x_test = np.array(test_data.data)
    y_test = np.array(test_data.target)

    if num_instances is not None:
        x_train = x_train[:num_instances]
        y_train = y_train[:num_instances]
        x_test = x_test[:num_instances]
        y_test = y_test[:num_instances]
    return (x_train, y_train), (x_test, y_test)


def build_graph():
    tf.keras.backend.clear_session()
    image_input = ak.ImageInput(shape=(32, 32, 3))
    image_input.batch_size = 32
    image_input.num_samples = 1000
    merged_outputs = ak.SpatialReduction()(image_input)
    head = ak.ClassificationHead(num_classes=10, shape=(10,))
    classification_outputs = head(merged_outputs)
    return ak.graph.Graph(inputs=image_input, outputs=classification_outputs)


def get_func_args(func):
    params = inspect.signature(func).parameters.keys()
    return set(params) - set(["self", "args", "kwargs"])
