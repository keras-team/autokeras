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
import pytest
import tensorflow as tf
from kerastuner.engine import hyperparameters

from autokeras.utils import utils


def test_validate_num_inputs_error():
    with pytest.raises(ValueError) as info:
        utils.validate_num_inputs([1, 2, 3], 2)

    assert "Expected 2 elements in the inputs list" in str(info.value)


def test_check_tf_version_error():
    utils.tf.__version__ = "2.1.0"

    with pytest.raises(ImportError) as info:
        utils.check_tf_version()

    assert "Tensorflow package version needs to be at least 2.3.0" in str(info.value)


def test_check_kt_version_error():
    utils.kerastuner.__version__ = "1.0.0"

    with pytest.raises(ImportError) as info:
        utils.check_kt_version()

    assert "Keras Tuner package version needs to be at least 1.0.2rc3" in str(
        info.value
    )


def test_run_with_adaptive_batch_size_raise_error():
    def func(**kwargs):
        raise tf.errors.ResourceExhaustedError(0, "", None)

    with pytest.raises(tf.errors.ResourceExhaustedError):
        utils.run_with_adaptive_batch_size(
            batch_size=64,
            func=func,
            x=tf.data.Dataset.from_tensor_slices(np.random.rand(100, 1)).batch(64),
            validation_data=tf.data.Dataset.from_tensor_slices(
                np.random.rand(100, 1)
            ).batch(64),
        )


def test_get_hyperparameter_with_none_return_hp():
    hp = utils.get_hyperparameter(None, hyperparameters.Choice("hp", [10, 20]), int)
    assert isinstance(hp, hyperparameters.Choice)


def test_get_hyperparameter_with_int_return_fixed():
    hp = utils.get_hyperparameter(10, hyperparameters.Choice("hp", [10, 20]), int)
    assert isinstance(hp, hyperparameters.Fixed)


def test_get_hyperparameter_with_hp_return_same():
    hp = utils.get_hyperparameter(
        hyperparameters.Choice("hp", [10, 30]),
        hyperparameters.Choice("hp", [10, 20]),
        int,
    )
    assert isinstance(hp, hyperparameters.Choice)
