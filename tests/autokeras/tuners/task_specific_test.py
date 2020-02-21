import copy

import kerastuner
import tensorflow as tf

import autokeras as ak
from autokeras import graph as graph_module
from autokeras.tuners import task_specific


def check_initial_hp(initial_hp, graph):
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(initial_hp)
    tf.keras.backend.clear_session()
    graph.build(hp)
    assert hp.values == initial_hp


def test_image_classifier_tuner0():
    input_node = ak.ImageInput(shape=(32, 32, 3))
    output_node = ak.ImageBlock()(input_node)
    output_node = ak.ClassificationHead(
        loss='categorical_crossentropy',
        output_shape=(10,))(output_node)
    graph = graph_module.Graph(input_node, output_node)
    check_initial_hp(task_specific.IMAGE_CLASSIFIER[0], graph)


def test_image_classifier_tuner1():
    input_node = ak.ImageInput(shape=(32, 32, 3))
    output_node = ak.ImageBlock()(input_node)
    output_node = ak.ClassificationHead(
        loss='categorical_crossentropy',
        output_shape=(10,))(output_node)
    graph = graph_module.Graph(input_node, output_node)
    check_initial_hp(task_specific.IMAGE_CLASSIFIER[1], graph)


def test_text_classifier_tuner0():
    input_node = ak.TextInput(shape=(1,))
    output_node = ak.TextBlock()(input_node)
    output_node = ak.ClassificationHead(
        loss='categorical_crossentropy',
        output_shape=(10,))(output_node)
    graph = graph_module.Graph(input_node, output_node)
    check_initial_hp(task_specific.TEXT_CLASSIFIER[0], graph)
