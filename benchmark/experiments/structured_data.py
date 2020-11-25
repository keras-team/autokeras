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

import numpy as np
import sklearn
import tensorflow as tf

import autokeras as ak
from benchmark.experiments import experiment


class StructuredDataClassifierExperiment(experiment.Experiment):
    def get_auto_model(self):
        return ak.StructuredDataClassifier(
            max_trials=10, directory=self.tmp_dir, overwrite=True
        )


class Titanic(StructuredDataClassifierExperiment):
    def __init__(self):
        super().__init__(name="Titanic")

    @staticmethod
    def load_data():
        TRAIN_DATA_URL = (
            "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
        )
        TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
        x_train = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
        x_test = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

        return (x_train, "survived"), (x_test, "survived")


class StructuredDataRegressorExperiment(experiment.Experiment):
    def get_auto_model(self):
        return ak.StructuredDataRegressor(
            max_trials=10, directory=self.tmp_dir, overwrite=True
        )


class CaliforniaHousing(StructuredDataRegressorExperiment):
    @staticmethod
    def load_data():
        house_dataset = sklearn.datasets.fetch_california_housing()
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            house_dataset.data,
            np.array(house_dataset.target),
            test_size=0.2,
            random_state=42,
        )
        return (x_train, y_train), (x_test, y_test)
