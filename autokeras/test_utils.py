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
from tensorflow import keras

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

TRAIN_CSV_PATH = keras.utils.get_file(
    fname=os.path.basename(TRAIN_DATA_URL), origin=TRAIN_DATA_URL
)
TEST_CSV_PATH = keras.utils.get_file(
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
    data = keras.utils.to_categorical(labels)
    if dtype == "np":
        return data
    if dtype == "dataset":
        return tf.data.Dataset.from_tensor_slices(data).batch(32)


def generate_text_data(num_instances=100):
    vocab = np.array(
        [
            ["adorable", "clueless", "dirty", "odd", "stupid"],
            ["puppy", "car", "rabbit", "girl", "monkey"],
            ["runs", "hits", "jumps", "drives", "barfs"],
            ["crazily.", "dutifully.", "foolishly.", "merrily.", "occasionally."],
        ]
    )
    return np.array(
        [
            " ".join([vocab[j][np.random.randint(0, 5)] for j in range(4)])
            for i in range(num_instances)
        ]
    )


def generate_data_with_categorical(
    num_instances=100, num_numerical=10, num_categorical=3, num_classes=5, dtype="np"
):
    categorical_data = np.random.randint(
        num_classes, size=(num_instances, num_categorical)
    )
    numerical_data = np.random.rand(num_instances, num_numerical)
    data = np.concatenate((numerical_data, categorical_data), axis=1)
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    if dtype == "np":
        return data
    if dtype == "dataset":
        return tf.data.Dataset.from_tensor_slices(data)


def build_graph():
    keras.backend.clear_session()
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


def get_object_detection_data():
    images = generate_data(num_instances=2, shape=(32, 32, 3))

    bbox_0 = np.random.rand(3, 4)
    class_id_0 = np.random.rand(
        3,
    )

    bbox_1 = np.random.rand(5, 4)
    class_id_1 = np.random.rand(
        5,
    )

    labels = np.array([(bbox_0, class_id_0), (bbox_1, class_id_1)], dtype=object)

    return images, labels
