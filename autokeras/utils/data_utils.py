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

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest


def batched(dataset):
    shape = nest.flatten(dataset_shape(dataset))[0]
    return len(shape) > 0 and shape[0] is None


def batch_dataset(dataset, batch_size):
    if batched(dataset):
        return dataset
    return dataset.batch(batch_size)


def split_dataset(dataset, validation_split):
    """Split dataset into training and validation.

    # Arguments
        dataset: tf.data.Dataset. The entire dataset to be split.
        validation_split: Float. The split ratio for the validation set.

    # Raises
        ValueError: If the dataset provided is too small to be split.

    # Returns
        A tuple of two tf.data.Dataset. The training set and the validation set.
    """
    num_instances = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    if num_instances < 2:
        raise ValueError(
            "The dataset should at least contain 2 batches to be split."
        )
    validation_set_size = min(
        max(int(num_instances * validation_split), 1), num_instances - 1
    )
    train_set_size = num_instances - validation_set_size
    train_dataset = dataset.take(train_set_size)
    validation_dataset = dataset.skip(train_set_size)
    return train_dataset, validation_dataset


def dataset_shape(dataset):
    return tf.compat.v1.data.get_output_shapes(dataset)


def unzip_dataset(dataset):
    return nest.flatten(
        [
            dataset.map(lambda *a: nest.flatten(a)[index])
            for index in range(len(nest.flatten(dataset_shape(dataset))))
        ]
    )


def cast_to_string(tensor):
    if tensor.dtype == tf.string:
        return tensor
    return tf.strings.as_string(tensor)


def cast_to_float32(tensor):
    if tensor.dtype == tf.float32:
        return tensor
    if tensor.dtype == tf.string:
        return tf.strings.to_number(tensor, tf.float32)
    return tf.cast(tensor, tf.float32)
