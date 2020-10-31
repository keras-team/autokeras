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

import re

import kerastuner
import tensorflow as tf
from kerastuner.engine import hyperparameters
from packaging.version import parse
from tensorflow.python.util import nest


def validate_num_inputs(inputs, num):
    inputs = nest.flatten(inputs)
    if not len(inputs) == num:
        raise ValueError(
            "Expected {num} elements in the inputs list "
            "but received {len} inputs.".format(num=num, len=len(inputs))
        )


def to_snake_case(name):
    intermediate = re.sub("(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    insecure = re.sub("([a-z])([A-Z])", r"\1_\2", intermediate).lower()
    return insecure


def check_tf_version() -> None:
    if parse(tf.__version__) < parse("2.3.0"):
        raise ImportError(
            "The Tensorflow package version needs to be at least 2.3.0 \n"
            "for AutoKeras to run. Currently, your TensorFlow version is \n"
            "{version}. Please upgrade with \n"
            "`$ pip install --upgrade tensorflow`. \n"
            "You can use `pip freeze` to check afterwards that everything is "
            "ok.".format(version=tf.__version__)
        )


def check_kt_version() -> None:
    if parse(kerastuner.__version__) < parse("1.0.2rc3"):
        raise ImportError(
            "The Keras Tuner package version needs to be at least 1.0.2rc3 \n"
            "for AutoKeras to run. Currently, your Keras Tuner version is \n"
            "{version}. Please upgrade with \n"
            "`$ pip install "
            "git+https://github.com/keras-team/keras-tuner.git@1.0.2rc3`. \n"
            "You can use `pip freeze` to check afterwards that everything is "
            "ok.".format(version=kerastuner.__version__)
        )


def contain_instance(instance_list, instance_type):
    return any([isinstance(instance, instance_type) for instance in instance_list])


def evaluate_with_adaptive_batch_size(model, batch_size, **fit_kwargs):
    return run_with_adaptive_batch_size(
        batch_size,
        lambda x, validation_data, **kwargs: model.evaluate(x, **kwargs),
        **fit_kwargs
    )


def predict_with_adaptive_batch_size(model, batch_size, **fit_kwargs):
    return run_with_adaptive_batch_size(
        batch_size,
        lambda x, validation_data, **kwargs: model.predict(x, **kwargs),
        **fit_kwargs
    )


def fit_with_adaptive_batch_size(model, batch_size, **fit_kwargs):
    history = run_with_adaptive_batch_size(
        batch_size, lambda **kwargs: model.fit(**kwargs), **fit_kwargs
    )
    return model, history


def run_with_adaptive_batch_size(batch_size, func, **fit_kwargs):
    x = fit_kwargs.pop("x")
    validation_data = None
    if "validation_data" in fit_kwargs:
        validation_data = fit_kwargs.pop("validation_data")
    while batch_size > 0:
        try:
            history = func(x=x, validation_data=validation_data, **fit_kwargs)
            break
        except tf.errors.ResourceExhaustedError as e:
            if batch_size == 1:
                raise e
            batch_size //= 2
            print(
                "Not enough memory, reduce batch size to {batch_size}.".format(
                    batch_size=batch_size
                )
            )
            x = x.unbatch().batch(batch_size)
            if validation_data is not None:
                validation_data = validation_data.unbatch().batch(batch_size)
    return history


def get_hyperparameter(value, hp, dtype):
    if value is None:
        return hp
    elif isinstance(value, dtype):
        return hyperparameters.Fixed(hp.name, value)
    return value


def add_to_hp(hp, hps, name=None):
    """Add the HyperParameter (self) to the HyperParameters.

    # Arguments
        hp: kerastuner.HyperParameters.
        name: String. If left unspecified, the hp name is used.
    """
    kwargs = hp.get_config()
    if name is None:
        name = hp.name
    kwargs.pop("conditions")
    kwargs.pop("name")
    class_name = hp.__class__.__name__
    func = getattr(hps, class_name)
    return func(name=name, **kwargs)
