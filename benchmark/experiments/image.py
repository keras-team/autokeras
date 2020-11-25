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

from tensorflow.keras import datasets

import autokeras as ak
from benchmark.experiments import experiment


class ImageClassifierExperiment(experiment.Experiment):
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
