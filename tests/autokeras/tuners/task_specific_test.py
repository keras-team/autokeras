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

import copy

import kerastuner

import autokeras as ak
from autokeras.tuners import task_specific


def test_img_clf_init_hp0_equals_hp_of_a_model(tmp_path):
    clf = ak.ImageClassifier(directory=tmp_path)
    clf.inputs[0].shape = (32, 32, 3)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.IMAGE_CLASSIFIER[0]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)

    assert set(init_hp.keys()) == set(hp._hps.keys())


def test_img_clf_init_hp1_equals_hp_of_a_model(tmp_path):
    clf = ak.ImageClassifier(directory=tmp_path)
    clf.inputs[0].shape = (32, 32, 3)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.IMAGE_CLASSIFIER[1]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)

    assert set(init_hp.keys()) == set(hp._hps.keys())


def test_img_clf_init_hp2_equals_hp_of_a_model(tmp_path):
    clf = ak.ImageClassifier(directory=tmp_path)
    clf.inputs[0].shape = (32, 32, 3)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.IMAGE_CLASSIFIER[2]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)

    assert set(init_hp.keys()) == set(hp._hps.keys())


def test_txt_clf_init_hp2_equals_hp_of_a_model(tmp_path):
    clf = ak.TextClassifier(directory=tmp_path)
    clf.inputs[0].shape = (1,)
    clf.inputs[0].batch_size = 6
    clf.inputs[0].num_samples = 1000
    clf.outputs[0].in_blocks[0].shape = (10,)
    clf.tuner.hypermodel.hypermodel.epochs = 1000
    clf.tuner.hypermodel.hypermodel.num_samples = 20000
    init_hp = task_specific.TEXT_CLASSIFIER[2]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)

    assert set(init_hp.keys()) == set(hp._hps.keys())


def test_txt_clf_init_hp1_equals_hp_of_a_model(tmp_path):
    clf = ak.TextClassifier(directory=tmp_path)
    clf.inputs[0].shape = (1,)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.TEXT_CLASSIFIER[1]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)

    assert set(init_hp.keys()) == set(hp._hps.keys())


def test_txt_clf_init_hp0_equals_hp_of_a_model(tmp_path):
    clf = ak.TextClassifier(directory=tmp_path)
    clf.inputs[0].shape = (1,)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.TEXT_CLASSIFIER[0]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)
    assert set(init_hp.keys()) == set(hp._hps.keys())


def test_sd_clf_init_hp0_equals_hp_of_a_model(tmp_path):
    clf = ak.StructuredDataClassifier(
        directory=tmp_path,
        column_names=["a", "b"],
        column_types={"a": "numerical", "b": "numerical"},
    )
    clf.inputs[0].shape = (2,)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.STRUCTURED_DATA_CLASSIFIER[0]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)

    assert set(init_hp.keys()) == set(hp._hps.keys())


def test_sd_reg_init_hp0_equals_hp_of_a_model(tmp_path):
    clf = ak.StructuredDataRegressor(
        directory=tmp_path,
        column_names=["a", "b"],
        column_types={"a": "numerical", "b": "numerical"},
    )
    clf.inputs[0].shape = (2,)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.STRUCTURED_DATA_REGRESSOR[0]
    hp = kerastuner.HyperParameters()
    hp.values = copy.copy(init_hp)

    clf.tuner.hypermodel.build(hp)

    assert set(init_hp.keys()) == set(hp._hps.keys())
