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

import autokeras as ak
from benchmark.experiments import experiment


class IMDB(experiment.Experiment):
    def __init__(self):
        super().__init__(name="IMDB")

    def get_auto_model(self):
        return ak.TextClassifier(
            max_trials=10, directory=self.tmp_dir, overwrite=True
        )

    @staticmethod
    def load_data():
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
        return (x_train, y_train), (x_test, y_test)
