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

import os
import shutil

import pytest
import tensorflow as tf

from autokeras.utils import io_utils
from tests import utils

IMG_DATA_DIR = os.path.join(
    os.path.dirname(
        tf.keras.utils.get_file(
            origin="https://storage.googleapis.com/"
            + "download.tensorflow.org/example_images/flower_photos.tgz",
            fname="image_data",
            extract=True,
        )
    ),
    "flower_photos",
)


def test_load_imdb_dataset():
    data_dir = os.path.join(
        os.path.dirname(
            tf.keras.utils.get_file(
                fname="text_data",
                origin="http://ai.stanford.edu/"
                + "~amaas/data/sentiment/aclImdb_v1.tar.gz",
                extract=True,
            )
        ),
        "aclImdb",
    )
    shutil.rmtree(os.path.join(data_dir, "train/unsup"))

    dataset = io_utils.text_dataset_from_directory(
        os.path.join(data_dir, "train"), max_length=20
    )

    for data in dataset:
        assert data[0].dtype == tf.string
        assert data[1].dtype == tf.string
        break


def test_load_image_data():
    dataset = io_utils.image_dataset_from_directory(
        IMG_DATA_DIR,
        image_size=(180, 180),
        validation_split=0.2,
        subset="training",
        seed=utils.SEED,
    )

    val_dataset = io_utils.image_dataset_from_directory(
        IMG_DATA_DIR,
        image_size=(180, 180),
        validation_split=0.2,
        subset="validation",
        seed=utils.SEED,
    )

    for data in dataset:
        assert data[0].numpy().shape == (32, 180, 180, 3)
        assert data[1].dtype == tf.string
        break

    for data in val_dataset:
        assert data[0].numpy().shape == (32, 180, 180, 3)
        assert data[1].dtype == tf.string
        break


def test_load_image_data_raise_subset_error():
    with pytest.raises(ValueError) as info:
        io_utils.image_dataset_from_directory(
            IMG_DATA_DIR,
            image_size=(180, 180),
            validation_split=0.2,
            subset="abcd",
            seed=utils.SEED,
        )
    assert "`subset` must be either" in str(info.value)


def test_load_image_data_raise_color_mode_error():
    with pytest.raises(ValueError) as info:
        io_utils.image_dataset_from_directory(
            IMG_DATA_DIR, image_size=(180, 180), color_mode="abcd"
        )
    assert "`color_mode` must be one of" in str(info.value)


def test_load_image_data_rgba():
    io_utils.image_dataset_from_directory(
        IMG_DATA_DIR, image_size=(180, 180), color_mode="rgba"
    )


def test_load_image_data_grey_scale():
    io_utils.image_dataset_from_directory(
        IMG_DATA_DIR, image_size=(180, 180), color_mode="grayscale"
    )


def test_path_to_image():
    img_dir = os.path.join(IMG_DATA_DIR, "roses")
    assert isinstance(
        io_utils.path_to_image(
            os.path.join(img_dir, os.listdir(img_dir)[5]),
            num_channels=3,
            image_size=(180, 180),
            interpolation="bilinear",
        ),
        tf.Tensor,
    )
