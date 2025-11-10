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

import keras
import numpy as np
import tree
from keras import ops


def split_dataset(dataset, validation_split):
    """Split nested numpy arrays into training and validation.

    # Arguments
        dataset: A nested structure of numpy arrays.
        validation_split: Float. The split ratio for the validation set.

    # Raises
        ValueError: If the dataset provided is too small to be split.

    # Returns
        A tuple of two nested structures.
        The training set and the validation set.
    """
    # Get the number of instances from the first array's shape
    first_shape = tree.flatten(
        tree.map_structure(lambda x: np.array(x.shape), dataset)
    )[0]
    num_instances = first_shape[0]
    if num_instances < 2:
        raise ValueError(
            "The dataset should at least contain 2 instances to be split."
        )
    validation_set_size = min(
        max(int(num_instances * validation_split), 1), num_instances - 1
    )
    train_set_size = num_instances - validation_set_size
    # Split each array in the nested structure
    train_dataset = tree.map_structure(lambda x: x[:train_set_size], dataset)
    validation_dataset = tree.map_structure(
        lambda x: x[train_set_size:], dataset
    )
    return train_dataset, validation_dataset


def cast_to_float32(tensor):
    if keras.backend.standardize_dtype(tensor.dtype) == "float32":
        return tensor
    return ops.cast(tensor, "float32")  # pragma: no cover


def dataset_shape(dataset):
    """Recursively get shapes from a nested structure of numpy arrays.

    # Arguments
        nested_arrays: A nested structure (dict, list, tuple) containing numpy
            arrays.

    # Returns
        The same nested structure with shapes instead of arrays.
    """
    return tree.map_structure(lambda x: np.array(x.shape), dataset)
