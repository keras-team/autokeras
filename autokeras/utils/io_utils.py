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

import json

from keras_tuner.engine import hyperparameters

WHITELIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def deserialize_block_arg(arg):
    if isinstance(arg, dict):
        return hyperparameters.deserialize(arg)
    return arg


def serialize_block_arg(arg):
    if isinstance(arg, hyperparameters.HyperParameter):
        return hyperparameters.serialize(arg)
    return arg
