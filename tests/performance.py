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

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist

import autokeras as ak
from tests import utils


def test_mnist_accuracy_over_98(tmp_path):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    clf = ak.ImageClassifier(max_trials=1, directory=tmp_path)
    clf.fit(x_train, y_train, epochs=10)
    accuracy = clf.evaluate(x_test, y_test)[1]
    assert accuracy >= 0.98


def test_cifar10_accuracy_over_93(tmp_path):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    clf = ak.ImageClassifier(max_trials=2, directory=tmp_path)
    clf.fit(x_train, y_train, epochs=5)
    accuracy = clf.evaluate(x_test, y_test)[1]
    assert accuracy >= 0.93


def test_imdb_accuracy_over_84(tmp_path):
    (x_train, y_train), (x_test, y_test) = utils.imdb_raw(num_instances=None)
    clf = ak.TextClassifier(max_trials=2, directory=tmp_path)
    clf.fit(x_train, y_train, epochs=2)
    accuracy = clf.evaluate(x_test, y_test)[1]
    assert accuracy >= 0.84
