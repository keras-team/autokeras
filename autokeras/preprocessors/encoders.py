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

import numpy as np
import tensorflow as tf

from autokeras import keras_layers
from autokeras.engine import preprocessor


class Encoder(preprocessor.TargetPreprocessor):
    """Transform labels to encodings.

    # Arguments
        labels: A list of labels of any type. The labels to be encoded.
    """

    def __init__(self, labels, **kwargs):
        super().__init__(**kwargs)
        self.labels = [
            label.decode("utf-8") if isinstance(label, bytes) else str(label)
            for label in labels
        ]

    def get_config(self):
        return {"labels": self.labels}

    def fit(self, dataset):
        return

    def transform(self, dataset):
        """Transform labels to integer encodings.

        # Arguments
            dataset: tf.data.Dataset. The dataset to be transformed.

        # Returns
            tf.data.Dataset. The transformed dataset.
        """
        keys_tensor = tf.constant(self.labels)
        vals_tensor = tf.constant(list(range(len(self.labels))))
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1
        )

        return dataset.map(lambda x: table.lookup(tf.reshape(x, [-1])))


class OneHotEncoder(Encoder):
    def transform(self, dataset):
        """Transform labels to one-hot encodings.

        # Arguments
            dataset: tf.data.Dataset. The dataset to be transformed.

        # Returns
            tf.data.Dataset. The transformed dataset.
        """
        dataset = super().transform(dataset)
        eye = tf.eye(len(self.labels))
        dataset = dataset.map(lambda x: tf.nn.embedding_lookup(eye, x))
        return dataset

    def postprocess(self, data):
        """Transform probabilities back to labels.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The original labels.
        """
        return np.array(
            list(
                map(
                    lambda x: self.labels[x],
                    np.argmax(np.array(data), axis=1),
                )
            )
        ).reshape(-1, 1)


class LabelEncoder(Encoder):
    """Transform the labels to integer encodings."""

    def transform(self, dataset):
        """Transform labels to integer encodings.

        # Arguments
            dataset: tf.data.Dataset. The dataset to be transformed.

        # Returns
            tf.data.Dataset. The transformed dataset.
        """
        dataset = super().transform(dataset)
        dataset = dataset.map(lambda x: tf.expand_dims(x, axis=-1))
        return dataset

    def postprocess(self, data):
        """Transform probabilities back to labels.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The original labels.
        """
        return np.array(
            list(map(lambda x: self.labels[int(round(x[0]))], np.array(data)))
        ).reshape(-1, 1)


class Encoder(preprocessor.TargetPreprocessor):
    """Transform labels to encodings.

    # Arguments
        labels: A list of labels of any type. The labels to be encoded.
    """

    def __init__(self, labels, **kwargs):
        super().__init__(**kwargs)
        self.labels = [
            label.decode("utf-8") if isinstance(label, bytes) else str(label)
            for label in labels
        ]

    def get_config(self):
        return {"labels": self.labels}

    def fit(self, dataset):
        return

    def transform(self, dataset):
        """Transform labels to integer encodings.

        # Arguments
            dataset: tf.data.Dataset. The dataset to be transformed.

        # Returns
            tf.data.Dataset. The transformed dataset.
        """
        keys_tensor = tf.constant(self.labels)
        vals_tensor = tf.constant(list(range(len(self.labels))))
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1
        )

        return dataset.map(lambda x: table.lookup(tf.reshape(x, [-1])))


# class ObjectDetectionLabelEncoder(preprocessor.Preprocessor):
#     """Transform labels to encodings.
#
#     # Arguments
#         labels: A list of labels of any type. The labels to be encoded.
#     """
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         # self.labels = [
#         #     label.decode("utf-8") if isinstance(label, bytes) else str(label)
#         #     for label in labels
#         # ]
#         # Check if this is correct, could be done differently
#         self.preprocess_data = keras_layers.ObjectDetectionPreProcessing()  # 1st preprocessor
#         self.label_encoder = keras_layers.LabelEncoder()  # 2nd preprocessor
#
#     def get_config(self):
#         return {"label_encoder": self.label_encoder}
#
#     def fit(self, dataset):
#         return
#
#     def transform(self, dataset):
#         """Transform dataset to target encodings with (image, label)
#
#         # Arguments
#             dataset: tf.data.Dataset. The dataset to be transformed.
#
#         # Returns
#             tf.data.Dataset. The transformed dataset.
#         """
#         autotune = tf.data.experimental.AUTOTUNE
#         for item in dataset:
#             print("image shape: ", tf.shape(item[0]))
#             print("image: ", item[0])
#             # cv2_imshow(item[0].numpy())
#             print("bbox: ", item[1][0])
#             # print(item[1][1])
#             print("labels: ", item[1][1])
#             break
#         train_dataset = dataset.map(
#             lambda x, y: self.preprocess_data.data_transform(x,y),
#             num_parallel_calls=autotune
#         )
#         # train_dataset = train_dataset.shuffle(8 * batch_size)
#         # train_dataset = train_dataset.padded_batch(
#         #     batch_size=batch_size, p
#         #     adding_values=(0.0, 1e-8, -1),
#         #     drop_remainder=True
#         # )
#         print("after preprocessing: ", train_dataset.element_spec)
#         train_dataset = train_dataset.map(
#             lambda x, y, z: self.label_encoder.encode_sample_func(x, y, z),
#             num_parallel_calls=autotune
#         )
#         return train_dataset

class ObjectDetectionLabelEncoder(preprocessor.Preprocessor):
    """ObjectDetectionPreProcessing layer.

        # Arguments
            max_sequence_length: maximum length of the sequences after vectorization.
        """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.tokenizer = tokenization.FullTokenizer(
        #     # vocab_file=constants.BERT_VOCAB_PATH,
        #     do_lower_case=True,
        # )
        # self.max_sequence_length = max_sequence_length

    def get_config(self):
        config = super().get_config()
        # config.update({"max_sequence_length": self.max_sequence_length})
        return config

    def build(self, input_shape):
        self.batch_size = input_shape

    def transform(self, dataset):
        """Applies preprocessing step to a batch of images and bboxes and their
        labels.

        Arguments:
          inputs: A batch of [image, (bbox, label)].

        Returns:
          images: List of Resized and padded images with random horizontal
            flipping applied.
          bboxes: List of Bounding boxes with the shape `(num_objects, 4)`
           where each box is of the format `[x, y, width, height]`.
          class_ids: List of tensors representing the class id of the objects, having
            shape `(num_objects,)`.
        """
        # images = []
        # bboxes = []
        # class_ids = []
        # for i in range(inputs.shape[0]):
        #     image, bbox, class_id = self.data_transform(inputs[i])
        #     images.append(image)
        #     bboxes.append(bbox)
        #     class_ids.append(class_id)
        # # check if this needs to be converted to tf.Dataset or not
        # images = tf.stack(images)
        # bboxes = tf.stack(bboxes)
        # class_ids = tf.stack(class_ids)
        # return images, bboxes, class_ids
        print("NEW CHANGE: ", dataset.element_spec)
        return dataset.map(data_transform)

# @staticmethod
def data_transform(sample):
    sample_x = sample[0]
    sample_y = sample[1]
    print("input to data_transform: ", tf.shape(sample_x), tf.shape(sample_y[0]), tf.shape(sample_y[1]))
    image = sample_x
    bbox = swap_xy(sample_y[0])  # check this function
    class_id = tf.cast(sample_y[1], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id

# @staticmethod
def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes

# @staticmethod
def resize_and_pad_image(
        image,
        min_side=800.0,
        max_side=1333.0,
        jitter=[640, 1024],
        stride=128.0,
):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio

# @staticmethod
def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    print("input to swapy_xy: ", tf.shape(boxes[0]))
    return tf.stack(
        [boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1
    )

# @staticmethod
def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [
            (boxes[..., :2] + boxes[..., 2:]) / 2.0,
            boxes[..., 2:] - boxes[..., :2],
        ],
        axis=-1,
    )