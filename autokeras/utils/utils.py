import json
import re

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
    if parse(tf.__version__) < parse('2.1.0'):
        raise ImportError(
            'The Tensorflow package version needs to be at least v2.1.0 \n'
            'for AutoKeras to run. Currently, your TensorFlow version is \n'
            'v{version}. Please upgrade with \n'
            '`$ pip install --upgrade tensorflow` -> GPU version \n'
            'or \n'
            '`$ pip install --upgrade tensorflow-cpu` -> CPU version. \n'
            'You can use `pip freeze` to check afterwards that everything is '
            'ok.'.format(version=tf.__version__)
        )


def save_json(path, obj):
    obj = json.dumps(obj)
    with tf.io.gfile.GFile(path, 'w') as f:
        f.write(obj)


def load_json(path):
    with tf.io.gfile.GFile(path, 'r') as f:
        obj = f.read()
    return json.loads(obj)


def adapt_model(model, dataset):
    # TODO: Remove this function after TF has fit-to-adapt feature.
    from tensorflow.keras.layers.experimental import preprocessing
    x = dataset.map(lambda x, y: x)

    def get_output_layer(tensor):
        tensor = nest.flatten(tensor)[0]
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            input_node = nest.flatten(layer.input)[0]
            if input_node is tensor:
                return layer
        return None

    for index, input_node in enumerate(nest.flatten(model.input)):
        def get_data(*args):
            return args[index]

        temp_x = x.map(get_data)
        layer = get_output_layer(input_node)
        while isinstance(layer, preprocessing.PreprocessingLayer):
            layer.adapt(temp_x)
            layer = get_output_layer(layer.output)
    return model
