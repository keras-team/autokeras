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

STATE = {}


class State:
    def __init__(self):
        self.registered_inputs = []
        self.registered_outputs = []

        # Passing y class info from preprocessor to postprocessor.
        self.y_info

    def register_inputs(self, inputs):
        raise NotImplementedError

    def register_outputs(self, inputs):
        # Remember to check duplication
        raise NotImplementedError


def get_state():
    return STATE[threading.get_ident()]


def init_state():
    STATE[threading.get_ident()] = State()
