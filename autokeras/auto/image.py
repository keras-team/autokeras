import tensorflow as tf

from autokeras import const
from autokeras.auto import auto_pipeline
from autokeras.auto import processor
from autokeras.hypermodel import hyper_block, hyper_node
from autokeras.hypermodel import hyper_head
from autokeras import tuner

def augmentedDataGenerator(dataset='cifar10'):
    # You give the dataset name, we return the augmented data!
    if dataset == 'cifar10':
        (x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset == 'cifar100':
        (x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    elif dataset == 'mnist':
        (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
        # Fasion-MNIST
        (x_train, y_train),(x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Augment the images
    for i in range(x_train.shape[0]):
        x_train[i] = augment_image(x_train[i])
    y_train = tf.keras.utils.to_categorical(y_train,num_classes=None)
    return (x_train, y_train)

def _get_min_and_max(value, name):
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


def augment_image(image,
                  rotation_range=0,  # either 0, 90, 180
                  horizontal_crop_range=0.,  # fraction 0-1
                  vertical_crop_range=0.,  # fraction 0-1
                  brightness_range=0.,  # fraction 0-1  [X]
                  saturation_range=0.,  # fraction 0-1  [X]
                  contrast_range=0.,  # fraction 0-1  [X]
                  horizontal_flip=False,  # boolean  [X]
                  vertical_flip=False,
                  scale_factor=0. , # scale factor
                  translation_top=0. ,
                  translation_bottome=0.,
                  translation_left=0.,
                  translation_right=0.,
                  Gaussian_noise=False):  # boolean  [X]
    target_height = image.shape[0]
    target_width = image.shape[1]
    if Gaussian_noise:
        # TODO: Add Gaussian noise to the image
        pass
    if scale_factor:
        # TODO: Scale the image using scikit-image
        pass
    if translation_bottome or translation_left or translation_right or translation_top:
        # TODO; Translate the images
        pass
    if rotation_range:
        if rotation_range == 90:
            k_choices = {0, 1, 3}
        elif rotation_range == 180:
            k_choices = {0, 1, 2, 3}
        # TODO
    if brightness_range:
        min_value, max_value = _get_min_and_max(
            brightness_range, 'brightness_range')
        image = tf.image.random_brightness(image, min_value, max_value)
 
    if saturation_range:
        min_value, max_value = _get_min_and_max(
            brightness_range, 'saturation_range')
        image = tf.image.random_saturation(x, min_value, max_value)

    if contrast_range:
        min_value, max_value = _get_min_and_max(
            contrast_range, 'contrast_range')
        image = tf.image.random_contrast(x, min_value, max_value)

    if horizontal_crop_range or vertical_crop_range:
        if image.shape.rank == 3:
            height = image.shape[1]
            width = image.shape[2]
        else:
            height = image.shape[0]
            width = image.shape[1]

        height_repartition_factor = random.random()
        crop_height = int(height * vertical_crop_range)
        offset_height = math.floor(height_repartition_factor * target_height)
        target_height = offset_height + crop_height

        width_repartition_factor = random.random()
        crop_width = int(width * horizontal_crop_range)
        offset_width = math.floor(width_repartition_factor * target_width)
        target_width = offset_width + crop_width

        image = tf.image.crop_to_bounding_box(
            image,
            offset_height,
            offset_width,
            target_height,
            target_width)

    if horizontal_flip:
        image = tf.image.random_flip_left_right(image)
    if vertical_flip:
        image = tf.image.random_flip_up_down(image)
    return image


class ImageTuner(tuner.SequentialRandomSearch):

    def _run(self, hyperparameters, fit_kwargs):
        # Build a model instance.
        model = self.hypermodel.build(hyperparameters)

        self._check_space(hyperparameters)
        
        (x_train, y_train) = augmentedDataGenerator('cifar100')
        
        history = model.fit(x_train,y_train,batch_size=32)
        metric_name = model.metrics_names[1]
        feedback = history.history['val_' + metric_name][-1]
        
        return model, feedback

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
