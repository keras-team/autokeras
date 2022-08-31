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
import multiprocessing
import os
from typing import Optional
from typing import Tuple

import numpy as np
import tensorflow as tf
from keras_tuner.engine import hyperparameters

WHITELIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def save_json(path, obj):
    obj = json.dumps(obj)
    with tf.io.gfile.GFile(path, "w") as f:
        f.write(obj)


def load_json(path):
    with tf.io.gfile.GFile(path, "r") as f:
        obj = f.read()
    return json.loads(obj)


def index_directory(
    directory,
    labels,
    formats,
    class_names=None,
    shuffle=True,
    seed=None,
    follow_links=False,
):
    """Make list of all files in the subdirs of `directory`, with their labels.

    # Arguments
      directory: The target directory (string).
      labels: Either "inferred"
          (labels are generated from the directory structure),
          None (no labels),
          or a list/tuple of integer labels of the same size as the number of
          valid files found in the directory. Labels should be sorted according
          to the alphanumeric order of the image file paths
          (obtained via `os.walk(directory)` in Python).
      formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").
      class_names: Only valid if "labels" is "inferred". This is the explicit
          list of class names (must match names of subdirectories). Used
          to control the order of the classes
          (otherwise alphanumerical order is used).
      shuffle: Whether to shuffle the data. Default: True.
          If set to False, sorts the data in alphanumeric order.
      seed: Optional random seed for shuffling.
      follow_links: Whether to visits subdirectories pointed to by symlinks.

    # Returns
      tuple (file_paths, labels, class_names).
        file_paths: list of file paths (strings).
        labels: list of matching integer labels (same length as file_paths)
        class_names: names of the classes corresponding to these labels, in order.
    """
    subdirs = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            subdirs.append(subdir)
    if not class_names:
        class_names = subdirs
    else:
        if set(class_names) != set(subdirs):  # pragma: no cover
            raise ValueError(  # pragma: no cover
                "The `class_names` passed did not match the "
                "names of the subdirectories of the target directory. "
                "Expected: %s, but received: %s" % (subdirs, class_names)
            )
    class_indices = dict(zip(class_names, range(len(class_names))))

    # Build an index of the files
    # in the different class subfolders.
    pool = multiprocessing.pool.ThreadPool()
    results = []
    filenames = []

    for dirpath in (os.path.join(directory, subdir) for subdir in subdirs):
        results.append(
            pool.apply_async(
                index_subdirectory, (dirpath, class_indices, follow_links, formats)
            )
        )
    labels_list = []
    for res in results:
        partial_filenames, partial_labels = res.get()
        labels_list.append(partial_labels)
        filenames += partial_filenames
    i = 0
    labels = np.zeros((len(filenames),), dtype="int32")
    for partial_labels in labels_list:
        labels[i : i + len(partial_labels)] = partial_labels
        i += len(partial_labels)

    print(
        "Found %d files belonging to %d classes."
        % (len(filenames), len(class_names))
    )
    pool.close()
    pool.join()
    file_paths = [os.path.join(directory, fname) for fname in filenames]

    if shuffle:
        # Shuffle globally to erase macro-structure
        if seed is None:
            seed = np.random.randint(1e6)  # pragma: no cover
        rng = np.random.RandomState(seed)
        rng.shuffle(file_paths)
        rng = np.random.RandomState(seed)
        rng.shuffle(labels)
    return file_paths, labels, class_names


def iter_valid_files(directory, follow_links, formats):
    walk = os.walk(directory, followlinks=follow_links)
    for root, _, files in sorted(walk, key=lambda x: x[0]):
        for fname in sorted(files):
            if fname.lower().endswith(formats):
                yield root, fname


def index_subdirectory(directory, class_indices, follow_links, formats):
    """Recursively walks directory and list image paths and their class index.

    # Arguments
      directory: string, target directory.
      class_indices: dict mapping class names to their index.
      follow_links: boolean, whether to recursively follow subdirectories
        (if False, we only list top-level images in `directory`).
      formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").

    # Returns
      tuple `(filenames, labels)`. `filenames` is a list of relative file
        paths, and `labels` is a list of integer labels corresponding to these
        files.
    """
    dirname = os.path.basename(directory)
    valid_files = iter_valid_files(directory, follow_links, formats)
    labels = []
    filenames = []
    for root, fname in valid_files:
        labels.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory)
        )
        filenames.append(relative_path)
    return filenames, labels


def get_training_or_validation_split(samples, labels, validation_split, subset):
    """Potentially restict samples & labels to a training or validation split.

    # Arguments
        samples: List of elements.
        labels: List of corresponding labels.
        validation_split: Float, fraction of data to reserve for validation.
        subset: Subset of the data to return.
            Either "training", "validation", or None.
            If None, we return all of the data.

    # Returns
        tuple (samples, labels), potentially restricted to the specified subset.
    """
    if not validation_split:
        return samples, labels

    num_val_samples = int(validation_split * len(samples))
    if subset == "training":
        print("Using %d files for training." % (len(samples) - num_val_samples,))
        samples = samples[:-num_val_samples]
        labels = labels[:-num_val_samples]
    elif subset == "validation":
        print("Using %d files for validation." % (num_val_samples,))
        samples = samples[-num_val_samples:]
        labels = labels[-num_val_samples:]
    else:
        raise ValueError(
            '`subset` must be either "training" '
            'or "validation", received: %s' % (subset,)
        )
    return samples, labels


def text_dataset_from_directory(
    directory: str,
    batch_size: int = 32,
    max_length: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    validation_split: Optional[float] = None,
    subset: Optional[str] = None,
) -> tf.data.Dataset:
    """Generates a `tf.data.Dataset` from text files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_text_1.txt
    ......a_text_2.txt
    ...class_b/
    ......b_text_1.txt
    ......b_text_2.txt
    ```

    Then calling `text_dataset_from_directory(main_directory)`
    will return a `tf.data.Dataset` that yields batches of texts from
    the subdirectories `class_a` and `class_b`, together with labels
    'class_a' and 'class_b'.

    Only `.txt` files are supported at this time.

    # Arguments
        directory: Directory where the data is located.
            If `labels` is "inferred", it should contain
            subdirectories, each containing text files for a class.
            Otherwise, the directory structure is ignored.
        batch_size: Size of the batches of data. Defaults to 32.
        max_length: Maximum size of a text string. Texts longer than this will
            be truncated to `max_length`.
        shuffle: Whether to shuffle the data. Default: True.
            If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: One of "training" or "validation".
            Only used if `validation_split` is set.

    # Returns
        A `tf.data.Dataset` object, which yields a tuple `(texts, labels)`,
            where both has shape `(batch_size,)` and type of tf.string.
    """
    if seed is None:
        seed = np.random.randint(1e6)
    file_paths, labels, class_names = index_directory(
        directory, "inferred", formats=(".txt",), shuffle=shuffle, seed=seed
    )

    file_paths, labels = get_training_or_validation_split(
        file_paths, labels, validation_split, subset
    )

    strings = tf.data.Dataset.from_tensor_slices(file_paths)
    strings = strings.map(tf.io.read_file)
    if max_length is not None:
        strings = strings.map(lambda x: tf.strings.substr(x, 0, max_length))

    labels = np.array(class_names)[np.array(labels)]
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((strings, labels))
    dataset = dataset.batch(batch_size)
    return dataset


def image_dataset_from_directory(
    directory: str,
    batch_size: int = 32,
    color_mode: str = "rgb",
    image_size: Tuple[int, int] = (256, 256),
    interpolation: str = "bilinear",
    shuffle: bool = True,
    seed: Optional[int] = None,
    validation_split: Optional[float] = None,
    subset: Optional[str] = None,
) -> tf.data.Dataset:
    """Generates a `tf.data.Dataset` from image files in a directory.
    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_image_1.jpg
    ......a_image_2.jpg
    ...class_b/
    ......b_image_1.jpg
    ......b_image_2.jpg
    ```

    Then calling `image_dataset_from_directory(main_directory)`
    will return a `tf.data.Dataset` that yields batches of images from
    the subdirectories `class_a` and `class_b`, together with labels
    'class_a' and 'class_b'.

    Supported image formats: jpeg, png, bmp, gif.
    Animated gifs are truncated to the first frame.

    # Arguments
        directory: Directory where the data is located.
            If `labels` is "inferred", it should contain
            subdirectories, each containing images for a class.
            Otherwise, the directory structure is ignored.
        batch_size: Size of the batches of data. Default: 32.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            Whether the images will be converted to
            have 1, 3, or 4 channels.
        image_size: Size to resize images to after they are read from disk.
            Defaults to `(256, 256)`.
            Since the pipeline processes batches of images that must all have
            the same size, this must be provided.
        interpolation: String, the interpolation method used when resizing images.
          Defaults to `bilinear`. Supports `bilinear`, `nearest`, `bicubic`,
          `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
        shuffle: Whether to shuffle the data. Default: True.
            If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: One of "training" or "validation".
            Only used if `validation_split` is set.

    # Returns
        A `tf.data.Dataset` object, which yields a tuple `(texts, labels)`,
        where `images` has shape `(batch_size, image_size[0], image_size[1],
        num_channels)` where `labels` has shape `(batch_size,)` and type of
        tf.string.
        - if `color_mode` is `grayscale`, there's 1 channel in the image
        tensors.
        - if `color_mode` is `rgb`, there are 3 channel in the image tensors.
        - if `color_mode` is `rgba`, there are 4 channel in the image tensors.
    """
    if color_mode == "rgb":
        num_channels = 3
    elif color_mode == "rgba":
        num_channels = 4
    elif color_mode == "grayscale":
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
            "Received: %s" % (color_mode,)
        )

    if seed is None:
        seed = np.random.randint(1e6)
    image_paths, labels, class_names = index_directory(
        directory, "inferred", formats=WHITELIST_FORMATS, shuffle=shuffle, seed=seed
    )

    image_paths, labels = get_training_or_validation_split(
        image_paths, labels, validation_split, subset
    )

    images = tf.data.Dataset.from_tensor_slices(image_paths)
    images = images.map(
        lambda img: path_to_image(img, num_channels, image_size, interpolation)
    )

    labels = np.array(class_names)[np.array(labels)]
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.batch(batch_size)
    return dataset


def path_to_image(image, num_channels, image_size, interpolation):
    image = tf.io.read_file(image)
    image = tf.io.decode_image(image, channels=num_channels, expand_animations=False)
    image = tf.image.resize(image, image_size, method=interpolation)
    image.set_shape((image_size[0], image_size[1], num_channels))
    return image


def deserialize_block_arg(arg):
    if isinstance(arg, dict):
        return hyperparameters.deserialize(arg)
    return arg


def serialize_block_arg(arg):
    if isinstance(arg, hyperparameters.HyperParameter):
        return hyperparameters.serialize(arg)
    return arg
