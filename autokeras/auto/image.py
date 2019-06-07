import tensorflow as tf

from autokeras import const
from autokeras.auto.auto_pipeline import AutoPipeline
from autokeras.auto.processor import Normalizer, OneHotEncoder
from autokeras.hypermodel.hyper_block import ImageBlock
from autokeras.hypermodel.hyper_head import ClassificationHead, RegressionHead
from autokeras.tuner import SequentialRandomSearch


class ImageTuner(SequentialRandomSearch):
    def _run(self, hyperparameters, fit_kwargs):
        # Build a model instance.
        model = self.hypermodel.build(hyperparameters)

        # Optionally disallow hyperparameters defined on the fly.
        old_space = hyperparameters.space[:]
        new_space = hyperparameters.space[:]
        if not self.allow_new_parameters and set(old_space) != set(new_space):
            diff = set(new_space) - set(old_space)
            raise RuntimeError(
                'The hypermodel has requested a parameter that was not part '
                'of `hyperparameters`, '
                'yet `allow_new_parameters` is set to False. '
                'The unknown parameters are: {diff}'.format(diff=diff))

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # rescale image pixels to [0, 1]
            rescale=None,  # 1. / self.max_val,
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False)

        datagen.fit(fit_kwargs['x'])
        fit_kwargs['batch_size'] = min(len(fit_kwargs['x']),
                                       fit_kwargs.get('batch_size',
                                                      default=const.Constant.BATCH_SIZE))
        data_flow = datagen.flow(fit_kwargs['x'],
                                 fit_kwargs['y'],
                                 fit_kwargs['batch_size'],
                                 shuffle=True)
        fit_kwargs.pop('x', None)
        fit_kwargs.pop('y', None)

        # Train model
        history = model.fit_generator(data_flow, **fit_kwargs)

        metric_name = model.metrics_names[1]
        feedback = history.history['val_' + metric_name][-1]
        return model, feedback


class ImageSupervised(AutoPipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_block = ImageBlock()
        self.head = None
        self.x_normalizer = Normalizer()

    def fit(self, x=None, y=None, **kwargs):
        self.tuner = ImageTuner(self, metrics=self.head.metrics)
        self.x_normalizer.fit(x)
        super().fit(x=self.x_normalizer.transform(x), y=y, **kwargs)

    def build(self, hp):
        output_node = self.image_block.build(hp, self.inputs)
        output_node = self.head.build(hp, output_node)
        model = tf.keras.Model(self.inputs, output_node)
        optimizer = hp.Choice('optimizer',
                              [tf.keras.optimizers.Adam,
                               tf.keras.optimizers.Adadelta,
                               tf.keras.optimizers.SGD])()

        model.compile(optimizer=optimizer,
                      metrics=self.head.metrics,
                      loss=self.head.loss)

        return model


class ImageClassifier(ImageSupervised):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = ClassificationHead()
        self.y_encoder = OneHotEncoder()

    def fit(self, x=None, y=None, **kwargs):
        self.y_encoder.fit(y)
        super().fit(x=x, y=self.y_encoder.transform(y), **kwargs)

    def predict(self, x, **kwargs):
        return self.y_encoder.inverse_transform(super().predict(x, **kwargs))


class ImageRegressor(ImageSupervised):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = RegressionHead()