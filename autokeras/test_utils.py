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

import keras
import numpy as np

import autokeras as ak
from autokeras import data

SEED = 5


def generate_data(num_instances=100, shape=(32, 32, 3)):
    np.random.seed(SEED)
    result = np.random.rand(*((num_instances,) + shape))
    if result.dtype == np.float64:
        result = result.astype(np.float32)
    return result


def generate_one_hot_labels(num_instances=100, num_classes=10, dtype="np"):
    np.random.seed(SEED)
    labels = np.random.randint(num_classes, size=num_instances)
    result = keras.utils.to_categorical(labels, num_classes=num_classes)
    if dtype == "np":
        return result
    if dtype == "dataset":
        return data.Dataset.from_tensor_slices(result).batch(32)


def generate_text_data(num_instances=100):
    vocab = np.array(
        [
            ["adorable", "clueless", "dirty", "odd", "stupid"],
            ["puppy", "car", "rabbit", "girl", "monkey"],
            ["runs", "hits", "jumps", "drives", "barfs"],
            [
                "crazily.",
                "dutifully.",
                "foolishly.",
                "merrily.",
                "occasionally.",
            ],
        ]
    )
    return np.array(
        [
            " ".join([vocab[j][np.random.randint(0, 5)] for j in range(4)])
            for i in range(num_instances)
        ]
    )


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

    labels = np.array(
        [(bbox_0, class_id_0), (bbox_1, class_id_1)], dtype=object
    )

    return images, labels
