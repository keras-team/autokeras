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


class Adapter(object):
    """Adpat the input and output format for Keras Model.

    Adapter is used by the input nodes and the heads of the hypermodel. It do
    some type checking for the data and converts it to compatible format.
    """

    def check(self, dataset):
        """Check if the dataset is valid for the input node.

        # Arguments
            dataset: numpy.ndarray. The dataset to be checked.

        # Returns
            Boolean. Whether the dataset is in compatible format.
        """
        return True

    def adapt(self, dataset):
        """Check, convert and batch the dataset.

        # Arguments
            dataset: Usually numpy.ndarray. The dataset to be converted.

        # Returns
            numpy.ndarray. The converted dataset.
        """
        self.check(dataset)
        return dataset
