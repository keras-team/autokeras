import tensorflow as tf

from autokeras import const
from autokeras.auto import auto_pipeline
from autokeras.auto import processor
from autokeras.hypermodel import hyper_block, hyper_node
from autokeras.hypermodel import hyper_head
from autokeras import tuner

def get_min_and_max(value, name):
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(
                    'Argument %s expected either a float between 0 and 1, '
                    'or a tuple of 2 floats between 0 and 1, but got: %s' % (value, name))
            min_value = value[0]
            max_value = value[1]
        else:
            min_value = 1. - value
            max_value = 1. + value
        return min_value, max_value

def augment_image(    x_train,
                      rotation_range=0,  # either 0, 90, 180, 270
                      random_crop_height=0,  # fraction 0-1
                      random_crop_width=0,  # fraction 0-1
                      random_crop_seed=0,   # positice number
                      brightness_range=0.,  # fraction 0-1  [X]
                      saturation_range=0.,  # fraction 0-1  [X]
                      contrast_range=0.,  # fraction 0-1  [X]
                      horizontal_flip=False,  # boolean  [X]
                      vertical_flip=False,
                      translation_top=0,  # translation_top are the number of pixels, so positive int type are required.
                      translation_bottom=0,
                      translation_left=0,
                      translation_right=0,
                      gaussian_noise=False):  # boolean  [X]
    x_train = tf.convert_to_tensor(x_train)
    length_dim = len(x_train.shape)
    if length_dim != 4:
        raise ValueError(
            'The input of x_train should be a [batch_size, height, width, channel] shape tensor or list, but we get %s' % (x_train.shape))
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
            image = tf.cast(image,dtype=tf.float32)
            if gaussian_noise:
                noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=1.0, dtype=tf.float32)
                image = tf.add(image, noise)

            if translation_bottom or translation_left or translation_right or translation_top:
                x = tf.image.pad_to_bounding_box(image, translation_top, translation_left,
                                                 target_height + translation_bottom + translation_top,
                                                 target_width + translation_right + translation_left)
                image = tf.image.crop_to_bounding_box(x, translation_bottom, translation_right, target_height,
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
                    saturation_range, 'saturation_range')
                print(min_value,max_value)
                image = tf.image.random_saturation(image, min_value, max_value)

            if contrast_range:
                min_value, max_value = get_min_and_max(
                    contrast_range, 'contrast_range')
                image = tf.image.random_contrast(image, min_value, max_value)

            if random_crop_height and random_crop_width:
                crop_size = [batch_num, random_crop_height, random_crop_width, channels]
                seed = np.random.randint(random_crop_seed)
                target_shape = (target_height,target_width)
                print(tf.random_crop(image, size=crop_size, seed=seed).shape)
                image = tf.image.resize_images(tf.random_crop(image, size=crop_size, seed=seed),size=target_shape)

            if horizontal_flip:
                image = tf.image.flip_left_right(image)

            if vertical_flip:
                image = tf.image.flip_up_down(image)
    return image

class SupervisedImagePipeline(auto_pipeline.AutoPipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_block = hyper_block.ImageBlock()
        self.head = None
        self.normalizer = processor.Normalizer()

    def fit(self, x=None, y=None, **kwargs):
        self.tuner = ImageTuner(self, metrics=self.head.metrics)
        self.normalizer.fit(x)
        self.inputs = [hyper_node.ImageInput()]
        super().fit(x=self.normalizer.transform(x), y=y, **kwargs)

    def build(self, hp, **kwargs):
        input_node = self.inputs[0].build(hp)
        output_node = self.image_block.build(hp, input_node)
        output_node = self.head.build(hp, output_node)
        model = tf.keras.Model(input_node, output_node)
        optimizer = hp.Choice('optimizer',
                              [tf.keras.optimizers.Adam,
                               tf.keras.optimizers.Adadelta,
                               tf.keras.optimizers.SGD])()

        model.compile(optimizer=optimizer,
                      loss=self.head.loss,
                      metrics=self.head.metrics)

        return model


class ImageClassifier(SupervisedImagePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = hyper_head.ClassificationHead()
        self.label_encoder = processor.OneHotEncoder()

    def fit(self, x=None, y=None, **kwargs):
        self.label_encoder.fit(y)
        self.head.output_shape = (self.label_encoder.num_classes,)
        super().fit(x=x, y=self.label_encoder.transform(y), **kwargs)

    def predict(self, x, **kwargs):
        return self.label_encoder.inverse_transform(super().predict(x, **kwargs))


class ImageRegressor(SupervisedImagePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = hyper_head.RegressionHead()

    def fit(self, x=None, y=None, **kwargs):
        self.head.output_shape = (1,)
        super().fit(x=x, y=y, **kwargs)

    def predict(self, x, **kwargs):
        return super().predict(x, **kwargs).flatten()
