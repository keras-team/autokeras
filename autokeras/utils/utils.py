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

import keras
import keras_tuner
import tree

# Collect OOM exceptions from available libraries
oom_exceptions = []
try:
    import tensorflow as tf

    oom_exceptions.append(tf.errors.ResourceExhaustedError)  # pragma: no cover
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

try:
    import torch

    oom_exceptions.append(torch.cuda.OutOfMemoryError)  # pragma: no cover
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

try:
    import jax

    oom_exceptions.append(jax.errors.ResourceExhaustedError)  # pragma: no cover
except (ImportError, AttributeError):  # pragma: no cover
    pass  # pragma: no cover

oom_exceptions = tuple(oom_exceptions)


def validate_num_inputs(inputs, num):
    inputs = tree.flatten(inputs)
    if not len(inputs) == num:
        raise ValueError(
            "Expected {num} elements in the inputs list "
            "but received {len} inputs.".format(num=num, len=len(inputs))
        )


def to_snake_case(name):
    intermediate = re.sub("(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    insecure = re.sub("([a-z])([A-Z])", r"\1_\2", intermediate).lower()
    return insecure


def contain_instance(instance_list, instance_type):
    return any(
        [isinstance(instance, instance_type) for instance in instance_list]
    )


def evaluate_with_adaptive_batch_size(
    model, batch_size, verbose=1, **fit_kwargs
):
    return run_with_adaptive_batch_size(
        batch_size,
        lambda x, validation_data, **kwargs: model.evaluate(
            x, verbose=verbose, **kwargs
        ),
        **fit_kwargs,
    )


def predict_with_adaptive_batch_size(
    model, batch_size, verbose=1, **fit_kwargs
):
    return run_with_adaptive_batch_size(
        batch_size,
        lambda x, validation_data, **kwargs: model.predict(
            x, verbose=verbose, **kwargs
        ),
        **fit_kwargs,
    )


def fit_with_adaptive_batch_size(model, batch_size, **fit_kwargs):
    history = run_with_adaptive_batch_size(
        batch_size, lambda **kwargs: model.fit(**kwargs), **fit_kwargs
    )
    return model, history


def run_with_adaptive_batch_size(batch_size, func, **fit_kwargs):
    validation_data = None
    if "validation_data" in fit_kwargs:
        validation_data = fit_kwargs.pop("validation_data")
    while batch_size > 0:
        try:
            history = func(
                validation_data=validation_data,
                batch_size=batch_size,
                **fit_kwargs,
            )
            break
        except oom_exceptions as e:  # pragma: no cover
            if batch_size == 1:  # pragma: no cover
                raise e  # pragma: no cover
            batch_size //= 2  # pragma: no cover
            print(  # pragma: no cover
                "Not enough memory, reduce batch size to {batch_size}.".format(
                    batch_size=batch_size
                )
            )
    return history


def get_hyperparameter(value, hp, dtype):
    if value is None:
        return hp
    return value


def add_to_hp(hp, hps, name=None):
    """Add the HyperParameter (self) to the HyperParameters.

    # Arguments
        hp: keras_tuner.HyperParameters.
        name: String. If left unspecified, the hp name is used.
    """
    if not isinstance(hp, keras_tuner.engine.hyperparameters.HyperParameter):
        return hp
    kwargs = hp.get_config()
    if name is None:
        name = hp.name
    kwargs.pop("conditions")
    kwargs.pop("name")
    class_name = hp.__class__.__name__
    func = getattr(hps, class_name)
    return func(name=name, **kwargs)


def serialize_keras_object(obj):
    return keras.utils.serialize_keras_object(obj)  # pragma: no cover


def deserialize_keras_object(config, module_objects=None, custom_objects=None):
    return keras.utils.deserialize_keras_object(
        config, custom_objects, module_objects
    )
