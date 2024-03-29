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

from autokeras.tuners import greedy

IMAGE_CLASSIFIER = [
    {
        "image_block_1/block_type": "vanilla",
        "image_block_1/normalize": True,
        "image_block_1/augment": False,
        "image_block_1/conv_block_1/kernel_size": 3,
        "image_block_1/conv_block_1/num_blocks": 1,
        "image_block_1/conv_block_1/num_layers": 2,
        "image_block_1/conv_block_1/max_pooling": True,
        "image_block_1/conv_block_1/separable": False,
        "image_block_1/conv_block_1/dropout": 0.25,
        "image_block_1/conv_block_1/filters_0_0": 32,
        "image_block_1/conv_block_1/filters_0_1": 64,
        "classification_head_1/spatial_reduction_1/reduction_type": "flatten",
        "classification_head_1/dropout": 0.5,
        "optimizer": "adam",
        "learning_rate": 1e-3,
    },
    {
        "image_block_1/block_type": "resnet",
        "image_block_1/normalize": True,
        "image_block_1/augment": True,
        "image_block_1/image_augmentation_1/horizontal_flip": True,
        "image_block_1/image_augmentation_1/vertical_flip": True,
        "image_block_1/image_augmentation_1/contrast_factor": 0.0,
        "image_block_1/image_augmentation_1/rotation_factor": 0.0,
        "image_block_1/image_augmentation_1/translation_factor": 0.1,
        "image_block_1/image_augmentation_1/zoom_factor": 0.0,
        "image_block_1/res_net_block_1/pretrained": False,
        "image_block_1/res_net_block_1/version": "resnet50",
        "image_block_1/res_net_block_1/imagenet_size": True,
        "classification_head_1/spatial_reduction_1/reduction_type": "global_avg",  # noqa: E501
        "classification_head_1/dropout": 0,
        "optimizer": "adam",
        "learning_rate": 1e-3,
    },
    {
        "image_block_1/block_type": "efficient",
        "image_block_1/normalize": True,
        "image_block_1/augment": True,
        "image_block_1/image_augmentation_1/horizontal_flip": True,
        "image_block_1/image_augmentation_1/vertical_flip": False,
        "image_block_1/image_augmentation_1/contrast_factor": 0.0,
        "image_block_1/image_augmentation_1/rotation_factor": 0.0,
        "image_block_1/image_augmentation_1/translation_factor": 0.1,
        "image_block_1/image_augmentation_1/zoom_factor": 0.0,
        "image_block_1/efficient_net_block_1/pretrained": True,
        "image_block_1/efficient_net_block_1/version": "b7",
        "image_block_1/efficient_net_block_1/trainable": True,
        "image_block_1/efficient_net_block_1/imagenet_size": True,
        "classification_head_1/spatial_reduction_1/reduction_type": "global_avg",  # noqa: E501
        "classification_head_1/dropout": 0,
        "optimizer": "adam",
        "learning_rate": 2e-5,
    },
]

TEXT_CLASSIFIER = [
    {
        "classification_head_1/dropout": 0,
        "optimizer": "adam_weight_decay",
        "learning_rate": 2e-5,
        "text_block_1/bert_block_1/max_sequence_length": 512,
    },
]


class ImageClassifierTuner(greedy.Greedy):
    def __init__(self, **kwargs):
        super().__init__(initial_hps=IMAGE_CLASSIFIER, **kwargs)


class TextClassifierTuner(greedy.Greedy):
    def __init__(self, **kwargs):
        super().__init__(initial_hps=TEXT_CLASSIFIER, **kwargs)
