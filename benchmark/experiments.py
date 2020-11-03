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
import shutil
import statistics
import timeit

import numpy as np
import sklearn
import tensorflow as tf
from sklearn.datasets import load_files
from tensorflow.keras import datasets

import autokeras as ak


class Experiment(object):
    def __init__(self, name, tmp_dir="tmp_dir"):
        self.name = name
        self.tmp_dir = tmp_dir

    def get_auto_model(self):
        raise NotImplementedError

    @staticmethod
    def load_data():
        raise NotImplementedError

    def run_once(self):
        (x_train, y_train), (x_test, y_test) = self.load_data()
        auto_model = self.get_auto_model()

        start_time = timeit.default_timer()
        auto_model.fit(x_train, y_train)
        stop_time = timeit.default_timer()

        accuracy = auto_model.evaluate(x_test, y_test)[1]
        total_time = stop_time - start_time

        return total_time, accuracy

    def run(self, repeat_times=1):
        total_times = []
        metric_values = []
        for i in range(repeat_times):
            total_time, metric = self.run_once()
            total_times.append(total_time)
            metric_values.append(metric)
            self.tear_down()
        return total_times, metric_values

    def tear_down(self):
        shutil.rmtree(self.tmp_dir)


class StructuredDataClassifierExperiment(Experiment):
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


class StructuredDataRegressorExperiment(Experiment):
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


class IMDB(Experiment):
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


class ImageClassifierExperiment(Experiment):
    def get_auto_model(self):
        return ak.ImageClassifier(
            max_trials=10, directory=self.tmp_dir, overwrite=True
        )


class MNIST(ImageClassifierExperiment):
    def __init__(self):
        super().__init__(name="MNIST")

    @staticmethod
    def load_data():
        return datasets.mnist.load_data()


class CIFAR10(ImageClassifierExperiment):
    def __init__(self):
        super().__init__(name="CIFAR10")

    @staticmethod
    def load_data():
        return datasets.cifar10.load_data()


def run_titanic():
    exp = Titanic()
    total_times, metric_values = exp.run(repeat_times=10)
    print("Average Time: {}".format(statistics.mean(total_times)))
    print("Average Accuracy: {}".format(statistics.mean(metric_values)))
    print("Accuracy Standard Deviation: {}".format(statistics.stdev(metric_values)))


if __name__ == "__main__":
    run_titanic()
