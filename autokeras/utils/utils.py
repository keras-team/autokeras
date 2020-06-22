import json
import re

import kerastuner
import tensorflow as tf
from packaging.version import parse
from tensorflow.python.util import nest


def validate_num_inputs(inputs, num):
    inputs = nest.flatten(inputs)
    if not len(inputs) == num:
        raise ValueError('Expected {num} elements in the inputs list '
                         'but received {len} inputs.'.format(num=num,
                                                             len=len(inputs)))


def get_name_scope():
    with tf.name_scope('a') as scope:
        name_scope = scope[:-2]
    return name_scope


def to_snake_case(name):
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure


def to_type_key(dictionary, convert_func):
    """Convert the keys of a dictionary to a different type.

    # Arguments
        dictionary: Dictionary. The dictionary to be converted.
        convert_func: Function. The function to convert a key.
    """
    return {convert_func(key): value
            for key, value in dictionary.items()}


def check_tf_version() -> None:
    if parse(tf.__version__) < parse('2.2.0'):
        raise ImportError(
            'The Tensorflow package version needs to be at least 2.2.0 \n'
            'for AutoKeras to run. Currently, your TensorFlow version is \n'
            '{version}. Please upgrade with \n'
            '`$ pip install --upgrade tensorflow`. \n'
            'You can use `pip freeze` to check afterwards that everything is '
            'ok.'.format(version=tf.__version__)
        )


def check_kt_version() -> None:
    if parse(kerastuner.__version__) < parse('1.0.2rc0'):
        raise ImportError(
            'The Keras Tuner package version needs to be at least 1.0.2rc0 \n'
            'for AutoKeras to run. Currently, your Keras Tuner version is \n'
            '{version}. Please upgrade with \n'
            '`$ pip install '
            'git+https://github.com/keras-team/keras-tuner.git@1.0.2rc0'
            '#egg=keras-tuner-1.0.2rc0`. \n'
            'You can use `pip freeze` to check afterwards that everything is '
            'ok.'.format(version=kerastuner.__version__)
        )


def save_json(path, obj):
    obj = json.dumps(obj)
    with tf.io.gfile.GFile(path, 'w') as f:
        f.write(obj)


def load_json(path):
    with tf.io.gfile.GFile(path, 'r') as f:
        obj = f.read()
    return json.loads(obj)


def contain_instance(instance_list, instance_type):
    return any([isinstance(instance, instance_type)
                for instance in instance_list])
