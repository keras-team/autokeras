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

from tensorflow.keras import layers


def get_global_average_pooling(shape):
    return [
        layers.GlobalAveragePooling1D,
        layers.GlobalAveragePooling2D,
        layers.GlobalAveragePooling3D,
    ][len(shape) - 3]


def get_global_max_pooling(shape):
    return [
        layers.GlobalMaxPool1D,
        layers.GlobalMaxPool2D,
        layers.GlobalMaxPool3D,
    ][len(shape) - 3]


def get_max_pooling(shape):
    return [
        layers.MaxPool1D,
        layers.MaxPool2D,
        layers.MaxPool3D,
    ][len(shape) - 3]


def get_conv(shape):
    return [layers.Conv1D, layers.Conv2D, layers.Conv3D][len(shape) - 3]


def get_sep_conv(shape):
    return [
        layers.SeparableConv1D,
        layers.SeparableConv2D,
        layers.Conv3D,
    ][len(shape) - 3]
