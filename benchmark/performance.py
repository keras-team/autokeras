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
import os

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist

import autokeras as ak


def imdb_raw(num_instances=100):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True,
    )

    # set path to dataset
    IMDB_DATADIR = os.path.join(os.path.dirname(dataset), "aclImdb")

    classes = ["pos", "neg"]
    train_data = load_files(
        os.path.join(IMDB_DATADIR, "train"), shuffle=True, categories=classes
    )
    test_data = load_files(
        os.path.join(IMDB_DATADIR, "test"), shuffle=False, categories=classes
    )

    x_train = np.array(train_data.data)
    y_train = np.array(train_data.target)
    x_test = np.array(test_data.data)
    y_test = np.array(test_data.target)

    if num_instances is not None:
        x_train = x_train[:num_instances]
        y_train = y_train[:num_instances]
        x_test = x_test[:num_instances]
        y_test = y_test[:num_instances]
    return (x_train, y_train), (x_test, y_test)


def test_mnist_accuracy_over_98(tmp_path):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    clf = ak.ImageClassifier(max_trials=1, directory=tmp_path)
    clf.fit(x_train, y_train, epochs=10)
    accuracy = clf.evaluate(x_test, y_test)[1]
    assert accuracy >= 0.98


def test_cifar10_accuracy_over_93(tmp_path):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    clf = ak.ImageClassifier(max_trials=3, directory=tmp_path)
    clf.fit(x_train, y_train, epochs=5)
    accuracy = clf.evaluate(x_test, y_test)[1]
    assert accuracy >= 0.93


def test_imdb_accuracy_over_92(tmp_path):
    (x_train, y_train), (x_test, y_test) = imdb_raw(num_instances=None)
    clf = ak.TextClassifier(max_trials=3, directory=tmp_path)
    clf.fit(x_train, y_train, batch_size=6, epochs=1)
    accuracy = clf.evaluate(x_test, y_test)[1]
    assert accuracy >= 0.92


def test_titaninc_accuracy_over_77(tmp_path):
    TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

    train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
    test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
    clf = ak.StructuredDataClassifier(max_trials=10, directory=tmp_path)

    clf.fit(train_file_path, "survived")

    accuracy = clf.evaluate(test_file_path, "survived")[1]
    assert accuracy >= 0.77
