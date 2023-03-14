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

from autokeras.prototype import base_block


class Preprocessor(base_block.BaseBlock):
    def _build_wrapper(self, hp, inputs, *args, **kwargs):
        # Accept only Dataset.
        # Return a Dataset.
        # Register ConcretePreprocessor.

        # How to register it to the right input node?

        # How do we register the ConcretePreprocessor?
        # Just get the return value of .build(). Register it to graph state
        # together with the input and output dataset.

        # What do we do when there are Preprocessors within Preprocessors?
        # We don't register all of them. Only register the outter most one.
        # It is more convenient to just have this one preprocessor to do all the
        # inside steps.
        # To judge if the current one is the outter most one, we need to use the
        # "with" statement to create a scope when a HyperModel.build() is
        # called. Record a stack of HyperModel, whose .build() is running. The
        # lower in the stack, the outter the HyperModel is.
        return super()._build_wrapper(hp, inputs, *args, **kwargs)

    def build(self, hp, dataset):
        # Should return a ConcretePreprocessor.
        pass
