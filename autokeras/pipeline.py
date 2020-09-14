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

import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import preprocessors as preprocessors_module
from autokeras.engine import hyper_preprocessor as hpps_module
from autokeras.engine import preprocessor as pps_module
from autokeras.utils import data_utils
from autokeras.utils import utils


class HyperPipeline(hpps_module.HyperPreprocessor):
    """A search space consists of HyperPreprocessors.

    # Arguments
        inputs: a list of lists of HyperPreprocessors.
        outputs: a list of lists of HyperPreprocessors.
    """

    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(**kwargs)
        self.inputs = inputs
        self.outputs = outputs

    @staticmethod
    def _build_preprocessors(hp, hpps_lists, dataset):
        sources = data_utils.unzip_dataset(dataset)
        preprocessors_list = []
        for source, hpps_list in zip(sources, hpps_lists):
            data = source
            preprocessors = []
            for hyper_preprocessor in hpps_list:
                preprocessor = hyper_preprocessor.build(hp, data)
                data = preprocessor.transform(data)
                preprocessors.append(preprocessor)
            preprocessors_list.append(preprocessors)
        return preprocessors_list

    def build(self, hp, dataset):
        """Build a Pipeline by Hyperparameters.

        # Arguments
            hp: Hyperparameters.
            dataset: tf.data.Dataset.

        # Returns
            An instance of Pipeline.
        """
        x = dataset.map(lambda x, y: x)
        y = dataset.map(lambda x, y: y)
        return Pipeline(
            inputs=self._build_preprocessors(hp, self.inputs, x),
            outputs=self._build_preprocessors(hp, self.outputs, y),
        )


def load_pipeline(filepath, custom_objects=None):
    """Load a Pipeline instance from disk."""
    if custom_objects is None:
        custom_objects = {}
    with tf.keras.utils.custom_object_scope(custom_objects):
        return Pipeline.from_config(utils.load_json(filepath))


class Pipeline(pps_module.Preprocessor):
    """A data pipeline for transform the entire dataset.

    # Arguments
        inputs: A list of lists of Preprocessors. For the input datasets for
            the model.
        outputs: A list of lists of Preprocessors. For the target datasets for
            the model.
    """

    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(**kwargs)
        self.inputs = inputs
        self.outputs = outputs

    def fit(self, dataset):
        """Fit the Preprocessors."""
        # TODO: Implement.
        return

    def transform(self, dataset):
        """Transform the dataset to be ready for the model.

        # Arguments
            dataset: tf.data.Dataset.

        # Returns
            An instance of tf.data.Dataset. The transformed dataset.
        """
        x = dataset.map(lambda x, y: x)
        y = dataset.map(lambda x, y: y)
        x = self.transform_x(x)
        y = self.transform_y(y)
        return tf.data.Dataset.zip((x, y))

    def transform_x(self, dataset):
        """Transform the input dataset for the model.

        # Arguments
            dataset: tf.data.Dataset. The input dataset for the model.

        # Returns
            An instance of tf.data.Dataset. The transformed dataset.
        """
        return self._transform_data(dataset, self.inputs)

    def transform_y(self, dataset):
        """Transform the target dataset for the model.

        # Arguments
            dataset: tf.data.Dataset. The target dataset for the model.

        # Returns
            An instance of tf.data.Dataset. The transformed dataset.
        """
        return self._transform_data(dataset, self.outputs)

    def _transform_data(self, dataset, pps_lists):
        sources = data_utils.unzip_dataset(dataset)
        transformed = []
        for pps_list, data in zip(pps_lists, sources):
            for preprocessor in pps_list:
                data = preprocessor.transform(data)
            transformed.append(data)
        if len(transformed) == 1:
            return transformed[0]
        return tuple(transformed)

    def save(self, filepath):
        utils.save_json(filepath, self.get_config())

    def get_config(self):
        return {
            "inputs": [
                [
                    preprocessors_module.serialize(preprocessor)
                    for preprocessor in preprocessors
                ]
                for preprocessors in self.inputs
            ],
            "outputs": [
                [
                    preprocessors_module.serialize(preprocessor)
                    for preprocessor in preprocessors
                ]
                for preprocessors in self.outputs
            ],
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            inputs=[
                [
                    preprocessors_module.deserialize(preprocessor)
                    for preprocessor in preprocessors
                ]
                for preprocessors in config["inputs"]
            ],
            outputs=[
                [
                    preprocessors_module.deserialize(preprocessor)
                    for preprocessor in preprocessors
                ]
                for preprocessors in config["outputs"]
            ],
        )

    def postprocess(self, y):
        """Postprocess the outputs of the model.

        # Arguments
            y: numpy.ndarray or a list of numpy.ndarrays. The output of the
                Keras model.

        # Returns
            A list or an instance of numpy.ndarray. The postprocessed data for
            the heads.
        """
        outputs = []
        for data, preprocessors in zip(nest.flatten(y), self.outputs):
            for preprocessor in preprocessors[::-1]:
                if isinstance(preprocessor, pps_module.TargetPreprocessor):
                    data = preprocessor.postprocess(data)
            outputs.append(data)
        if len(outputs) == 1:
            return outputs[0]
        return outputs
