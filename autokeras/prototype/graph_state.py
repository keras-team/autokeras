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

import threading

from tensorflow import keras

STATE = {}


class State:
    # It is a global accessible object to record all the useful information of a
    # specific build of the Graph.
    def __init__(self):
        self.inputs = []
        self.outputs = []

        # Passing y class info from preprocessor to postprocessor.
        self.y_info

    def register_inputs(self, inputs):
        raise NotImplementedError

    def register_outputs(self, inputs):
        # Remember to check duplication
        raise NotImplementedError

    def build_model(self):
        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs)
        return self.model


def get_state():
    return STATE[threading.get_ident()]


def init_state():
    state = State()
    STATE[threading.get_ident()] = state
    return state
