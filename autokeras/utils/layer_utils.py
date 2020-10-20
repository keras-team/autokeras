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


def get_global_average_pooling(shape):
    return [
        tf.keras.layers.GlobalAveragePooling1D,
        tf.keras.layers.GlobalAveragePooling2D,
        tf.keras.layers.GlobalAveragePooling3D,
    ][len(shape) - 3]


def get_global_max_pooling(shape):
    return [
        tf.keras.layers.GlobalMaxPool1D,
        tf.keras.layers.GlobalMaxPool2D,
        tf.keras.layers.GlobalMaxPool3D,
    ][len(shape) - 3]


def get_max_pooling(shape):
    return [
        tf.keras.layers.MaxPool1D,
        tf.keras.layers.MaxPool2D,
        tf.keras.layers.MaxPool3D,
    ][len(shape) - 3]


def get_conv(shape):
    return [tf.keras.layers.Conv1D, tf.keras.layers.Conv2D, tf.keras.layers.Conv3D][
        len(shape) - 3
    ]


def get_sep_conv(shape):
    return [
        tf.keras.layers.SeparableConv1D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.Conv3D,
    ][len(shape) - 3]
