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

from autokeras.engine import analyser

CATEGORICAL = "categorical"
NUMERICAL = "numerical"


class InputAnalyser(analyser.Analyser):
    def finalize(self):
        return


class ImageAnalyser(InputAnalyser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def finalize(self):
        if len(self.shape) not in [3, 4]:
            raise ValueError(
                "Expect the data to ImageInput to have shape (batch_size, "
                "height, width, channels) or (batch_size, height, width) "
                "dimensions, but got input shape {shape}".format(
                    shape=self.shape
                )
            )


class TextAnalyser(InputAnalyser):
    def correct_shape(self):
        if len(self.shape) == 1:
            return True
        return len(self.shape) == 2 and self.shape[1] == 1

    def finalize(self):
        if not self.correct_shape():
            raise ValueError(
                "Expect the data to TextInput to have shape "
                "(batch_size, 1), but "
                "got input shape {shape}.".format(shape=self.shape)
            )
        if not (self.dtype == "string"):
            raise TypeError(
                "Expect the data to TextInput to be strings, but got "
                "{type}.".format(type=self.dtype)
            )


class StructuredDataAnalyser(InputAnalyser):
    def __init__(self, column_names=None, column_types=None, **kwargs):
        super().__init__(**kwargs)
        self.column_names = column_names
        self.column_types = column_types
        # Variables for inferring column types.
        self.count_numerical = None
        self.count_categorical = None
        self.count_unique_numerical = []
        self.num_col = None

    def update(self, data):
        super().update(data)
        # The super class set self.dtype to "string" based on the input numpy
        # array.  However, the preprocessor will encode it to float32.  So, set
        # self.dtype to "float32", which would be propagated to the Keras Input
        # node.
        self.dtype = "float32"
        if len(self.shape) != 2:
            return
        # data is a numpy array containing all the data.
        self.num_col = data.shape[1]
        self.count_numerical = np.zeros(self.num_col)
        self.count_categorical = np.zeros(self.num_col)
        self.count_unique_numerical = [{} for _ in range(self.num_col)]
        for i in range(self.num_col):
            self._update_column(data[:, i], i)

    def _update_column(self, column_data, i):
        # Vectorized check for numerical values
        def is_numerical(x):
            try:
                float(x)
                return True
            except (ValueError, TypeError):
                return False

        numerical_mask = np.vectorize(is_numerical)(column_data)
        self.count_numerical[i] = np.sum(numerical_mask)
        self.count_categorical[i] = len(column_data) - np.sum(numerical_mask)

        if np.any(numerical_mask):
            # Get numerical values
            numerical_values = column_data[numerical_mask]
            # Handle bytes
            numerical_values = np.array(
                [
                    x.decode("utf-8") if isinstance(x, bytes) else x
                    for x in numerical_values
                ]
            )
            numerical_floats = numerical_values.astype(float)
            unique_vals, counts = np.unique(
                numerical_floats, return_counts=True
            )
            self.count_unique_numerical[i] = dict(zip(unique_vals, counts))

    def finalize(self):
        self.check()
        self.infer_column_types()

    def get_input_name(self):
        return "StructuredDataInput"

    def check(self):
        if len(self.shape) != 2:
            raise ValueError(
                "Expect the data to {input_name} to have shape "
                "(batch_size, num_features), but "
                "got input shape {shape}.".format(
                    input_name=self.get_input_name(), shape=self.shape
                )
            )

        # Fill in the column_names
        if self.column_names is None:
            if self.column_types:
                raise ValueError(
                    "column_names must be specified, if "
                    "column_types is specified."
                )
            self.column_names = [str(index) for index in range(self.shape[1])]

        # Check if column_names has the correct length.
        if len(self.column_names) != self.shape[1]:
            raise ValueError(
                "Expect column_names to have length {expect} "
                "but got {actual}.".format(
                    expect=self.shape[1], actual=len(self.column_names)
                )
            )

    def infer_column_types(self):
        column_types = {}

        for i in range(self.num_col):
            if self.count_categorical[i] > 0:
                column_types[self.column_names[i]] = CATEGORICAL
            elif (
                len(self.count_unique_numerical[i]) / self.count_numerical[i]
                < 0.05
            ):
                column_types[self.column_names[i]] = CATEGORICAL
            else:
                column_types[self.column_names[i]] = NUMERICAL

        # Partial column_types is provided.
        if self.column_types is None:
            self.column_types = {}
        for key, value in column_types.items():
            if key not in self.column_types:
                self.column_types[key] = value
