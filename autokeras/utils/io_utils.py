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

import tensorflow as tf


def save_json(path, obj):
    obj = json.dumps(obj)
    with tf.io.gfile.GFile(path, "w") as f:
        f.write(obj)


def load_json(path):
    with tf.io.gfile.GFile(path, "r") as f:
        obj = f.read()
    return json.loads(obj)
