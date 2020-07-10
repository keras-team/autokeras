import copy

import kerastuner
import pytest
import tensorflow as tf

import autokeras as ak
from autokeras.tuners import task_specific


@pytest.fixture
def clear_session():
    yield
    tf.keras.backend.clear_session()


def test_img_clf_init_hp0_equals_hp_of_a_model(clear_session, tmp_path):
    clf = ak.ImageClassifier(directory=tmp_path)
    clf.inputs[0].shape = (32, 32, 3)
    clf.outputs[0].in_blocks[0].output_shape = (10,)
    init_hp = task_specific.IMAGE_CLASSIFIER[0]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)

    assert set(init_hp.keys()) == set(hp._hps.keys())


def test_img_clf_init_hp1_equals_hp_of_a_model(clear_session, tmp_path):
    clf = ak.ImageClassifier(directory=tmp_path)
    clf.inputs[0].shape = (32, 32, 3)
    clf.outputs[0].in_blocks[0].output_shape = (10,)
    init_hp = task_specific.IMAGE_CLASSIFIER[1]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)

    assert set(init_hp.keys()) == set(hp._hps.keys())


def test_img_clf_init_hp2_equals_hp_of_a_model(clear_session, tmp_path):
    clf = ak.ImageClassifier(directory=tmp_path)
    clf.inputs[0].shape = (32, 32, 3)
    clf.outputs[0].in_blocks[0].output_shape = (10,)
    init_hp = task_specific.IMAGE_CLASSIFIER[2]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)

    assert set(init_hp.keys()) == set(hp._hps.keys())


def test_txt_clf_init_hp0_equals_hp_of_a_model(clear_session, tmp_path):
    clf = ak.TextClassifier(directory=tmp_path)
    clf.inputs[0].shape = (1,)
    clf.outputs[0].in_blocks[0].output_shape = (10,)
    init_hp = task_specific.TEXT_CLASSIFIER[0]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)

    assert set(init_hp.keys()) == set(hp._hps.keys())


def test_txt_clf_init_hp1_equals_hp_of_a_model(clear_session, tmp_path):
    clf = ak.TextClassifier(directory=tmp_path)
    clf.inputs[0].shape = (1,)
    clf.outputs[0].in_blocks[0].output_shape = (10,)
    init_hp = task_specific.TEXT_CLASSIFIER[1]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)

    assert set(init_hp.keys()) == set(hp._hps.keys())
