import copy

import kerastuner
import tensorflow as tf

import autokeras as ak
from autokeras import graph as graph_module
from autokeras.tuners import greedy
from autokeras.tuners import task_specific


def check_initial_hp(initial_hp, graph):
    hp = kerastuner.HyperParameters()
    for i in range(3):
        hp.values = copy.copy(initial_hp)
        graph.build(hp)
    assert len(set(initial_hp.keys()) - set(hp._hps.keys())) == 0


def test_image_classifier_tuner0():
    tf.keras.backend.clear_session()
    input_node = ak.ImageInput(shape=(32, 32, 3))
    output_node = ak.ImageBlock()(input_node)
    output_node = ak.ClassificationHead(
        loss='categorical_crossentropy',
        output_shape=(10,))(output_node)
    graph = graph_module.Graph(input_node, output_node)
    check_initial_hp(task_specific.IMAGE_CLASSIFIER[0], graph)


def test_image_classifier_tuner1():
    tf.keras.backend.clear_session()
    input_node = ak.ImageInput(shape=(32, 32, 3))
    output_node = ak.ImageBlock()(input_node)
    output_node = ak.ClassificationHead(
        loss='categorical_crossentropy',
        output_shape=(10,))(output_node)
    graph = graph_module.Graph(input_node, output_node)
    check_initial_hp(task_specific.IMAGE_CLASSIFIER[1], graph)


def test_text_classifier_tuner0():
    tf.keras.backend.clear_session()
    input_node = ak.TextInput(shape=(1,))
    output_node = ak.TextBlock()(input_node)
    output_node = ak.ClassificationHead(
        loss='categorical_crossentropy',
        output_shape=(10,))(output_node)
    graph = graph_module.Graph(input_node, output_node)
    check_initial_hp(task_specific.TEXT_CLASSIFIER[0], graph)


def test_image_classifier_oracle():
    tf.keras.backend.clear_session()
    input_node = ak.ImageInput(shape=(32, 32, 3))
    output_node = ak.ImageBlock()(input_node)
    output_node = ak.ClassificationHead(
        loss='categorical_crossentropy',
        output_shape=(10,))(output_node)
    graph = graph_module.Graph(input_node, output_node)
    oracle = greedy.GreedyOracle(
        hypermodel=graph,
        initial_hps=task_specific.IMAGE_CLASSIFIER,
        objective='val_loss')
    oracle._populate_space('0')
    hp = oracle.get_space()
    hp.values = task_specific.IMAGE_CLASSIFIER[0]
    assert len(set(
        task_specific.IMAGE_CLASSIFIER[0].keys()
    ) - set(
        oracle.get_space().values.keys())) == 0
    oracle._populate_space('1')
    assert len(set(
        task_specific.IMAGE_CLASSIFIER[1].keys()
    ) - set(
        oracle.get_space().values.keys())) == 0
