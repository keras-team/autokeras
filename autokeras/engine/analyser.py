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


class Analyser(object):
    """Analyze the dataset for useful information.

    Analyser is used by the input nodes and the heads of the hypermodel.  It
    analyzes the dataset to get useful information, e.g., the shape of the
    data, the data type of the dataset. The information will be used by the
    input nodes and heads to construct the data pipeline and to build the Keras
    Model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shape = None
        self.dtype = None
        self.num_samples = 0
        self.batch_size = None

    def update(self, data):
        """Update the statistics with a batch of data.

        # Arguments
            data: tf.Tensor. One batch of data from tf.data.Dataset.
        """
        if self.dtype is None:
            self.dtype = data.dtype
        if self.shape is None:
            self.shape = data.shape.as_list()
        if self.batch_size is None:
            self.batch_size = data.shape.as_list()[0]
        self.num_samples += data.shape.as_list()[0]

    def finalize(self):
        """Process recorded information after all updates."""
        raise NotImplementedError
