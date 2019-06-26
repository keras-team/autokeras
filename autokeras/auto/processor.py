import numpy as np
import tensorflow as tf


class OneHotEncoder(object):
    """A class that can format data.

    This class provides ways to transform data's classification label into
    vector.

    # Attributes
        data: The input data
        num_classes: The number of classes in the classification problem.
        labels: The number of labels.
        label_to_vec: Mapping from label to vector.
        int_to_label: Mapping from int to label.
    """

    def __init__(self):
        """Initialize a OneHotEncoder"""
        self.data = None
        self.num_classes = 0
        self.labels = None
        self.label_to_vec = {}
        self.int_to_label = {}

    def fit(self, data):
        """Create mapping from label to vector, and vector to label."""
        data = np.array(data).flatten()
        self.labels = set(data)
        self.num_classes = len(self.labels)
        for index, label in enumerate(self.labels):
            vec = np.array([0] * self.num_classes)
            vec[index] = 1
            self.label_to_vec[label] = vec
            self.int_to_label[index] = label

    def transform(self, data):
        """Get vector for every element in the data array."""
        data = np.array(data)
        if len(data.shape) > 1:
            data = data.flatten()
        return np.array(list(map(lambda x: self.label_to_vec[x], data)))

    def inverse_transform(self, data):
        """Get label for every element in data."""
        return np.array(list(map(lambda x: self.int_to_label[x],
                                 np.argmax(np.array(data), axis=1))))


class Normalizer(object):
    """ Perform basic image transformation and augmentation.

    # Attributes
        max_val: the maximum value of all data.
        mean: the mean value.
        std: the standard deviation.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=(0, 1, 2), keepdims=True).flatten()
        self.std = np.std(data, axis=(0, 1, 2), keepdims=True).flatten()

    def transform(self, data):
        """ Transform the test data, perform normalization.

        # Arguments
            data: Numpy array. The data to be transformed.

        # Returns
            A DataLoader instance.
        """
        # channel-wise normalize the image
        data = (data - self.mean) / self.std
        return data

    def get_min_and_max(value, name):
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(
                    'Argument %s expected either a float between 0 and 1, '
                    'or a tuple of 2 floats between 0 and 1
                    , but got: %s' % (value, name))
            min_value = value[0]
            max_value = value[1]
        else:
            min_value = 1. - value
            max_value = 1. + value
        return min_value, max_value
    
    def augment_image(self,
                      x_train,
                      rotation_range=0,  # either 0, 90, 180, 270
                      random_crop_height=0,  # fraction 0-1
                      random_crop_width=0,  # fraction 0-1
                      random_crop_seed=0,   # positice number
                      brightness_range=0.,  # fraction 0-1  [X]
                      saturation_range=0.,  # fraction 0-1  [X]
                      contrast_range=0.,  # fraction 0-1  [X]
                      horizontal_flip=False,  # boolean  [X]
                      vertical_flip=False,
                      translation_top=0,  # translation_top are the number of pixels.
                      translation_bottom=0,
                      translation_left=0,
                      translation_right=0,
                      gaussian_noise=False):  # boolean  [X]
        x_train = tf.convert_to_tensor(x_train)
        length_dim = len(x_train.shape)
        if length_dim != 4:
            raise ValueError(
                'The input of x_train should be a [batch_size, height, width, channel] '
                'shape tensor or list, but we get %s' % x_train.shape)
        batch_num = x_train.shape[0]
        target_height = x_train.shape[1]
        target_width = x_train.shape[2]
        channels = x_train.shape[3]
        dataset = tf.data.Dataset.from_tensor_slices(x_train)
        dataset = dataset.batch(batch_size=batch_num)
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        with tf.Session() as sess:
            for i in range(1):
                batch = sess.run([one_element])
                image = tf.convert_to_tensor(batch[0])
                image = tf.cast(image, dtype=tf.float32)
                if gaussian_noise:
                    noise = tf.random_normal(shape=tf.shape(image),
                                            mean=0.0, stddev=1.0, dtype=tf.float32)
                    image = tf.add(image, noise)

                if translation_bottom or translation_left \
                        or translation_right or translation_top:
                    x = tf.image.pad_to_bounding_box(image, translation_top,
                                                 translation_left,
                                                 target_height
                                                 + translation_bottom
                                                 + translation_top,
                                                 target_width
                                                 + translation_right
                                                 + translation_left)
                    image = tf.image.crop_to_bounding_box(x, translation_bottom,
                                                      translation_right, target_height,
                                                      target_width)

                if rotation_range:
                    if rotation_range == 90:
                        image = tf.image.rot90(image, k=1)
                    elif rotation_range == 180:
                        image = tf.image.rot90(image, k=2)
                    elif rotation_range == 270:
                        image = tf.image.rot90(image, k=3)
                    else:
                        image = tf.image.rot90(image, k=4)

                if brightness_range:
                    min_value, max_value = get_min_and_max(
                        brightness_range, 'brightness_range')
                    image = tf.image.random_brightness(image, min_value, max_value)

                if saturation_range:
                    min_value, max_value = get_min_and_max(
                        saturation_range, 
                        'saturation_range')
                    print(min_value, max_value)
                    image = tf.image.random_saturation(image, min_value, max_value)

                if contrast_range:
                    min_value, max_value = get_min_and_max(
                        contrast_range,
                        'contrast_range')
                    image = tf.image.random_contrast(
                        image, min_value, max_value)

                if random_crop_height and random_crop_width:
                    crop_size = [batch_num, random_crop_height
                        , random_crop_width, channels]
                    seed = np.random.randint(random_crop_seed)
                    target_shape = (target_height,target_width)
                    print(tf.random_crop(image, size=crop_size, seed=seed).shape)
                    image = tf.image.resize_images(
                        tf.random_crop(image, size=crop_size, seed=seed),
                        size=target_shape)

                if horizontal_flip:
                    image = tf.image.flip_left_right(image)

                if vertical_flip:
                    image = tf.image.flip_up_down(image)
        return image
