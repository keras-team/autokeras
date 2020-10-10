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

from autokeras import preprocessors
from autokeras.preprocessors import encoders
from autokeras.utils import data_utils


def test_one_hot_encoder_deserialize_transforms_to_np():
    encoder = encoders.OneHotEncoder(["a", "b", "c"])
    encoder.fit(np.array(["a", "b", "a"]))

    encoder = preprocessors.deserialize(preprocessors.serialize(encoder))
    one_hot = encoder.transform(
        tf.data.Dataset.from_tensor_slices([["a"], ["c"], ["b"]]).batch(2)
    )

    for data in one_hot:
        assert data.shape[1:] == [3]


def test_one_hot_encoder_decode_to_same_string():
    encoder = encoders.OneHotEncoder(["a", "b", "c"])

    result = encoder.postprocess(np.eye(3))

    assert np.array_equal(result, np.array([["a"], ["b"], ["c"]]))


def test_label_encoder_decode_to_same_string():
    encoder = encoders.LabelEncoder(["a", "b"])

    result = encoder.postprocess([[0], [1]])

    assert np.array_equal(result, np.array([["a"], ["b"]]))


def test_label_encoder_encode_to_correct_shape():
    encoder = encoders.LabelEncoder(["a", "b"])
    dataset = tf.data.Dataset.from_tensor_slices([["a"], ["b"]]).batch(32)

    result = encoder.transform(dataset)

    assert data_utils.dataset_shape(result).as_list() == [None, 1]
