import tensorflow as tf

from autokeras import const
from autokeras.auto import auto_pipeline
from autokeras.auto import processor
from autokeras.hypermodel import hyper_block, hyper_node
from autokeras.hypermodel import hyper_head
from autokeras import tuner


class ImageTuner(tuner.SequentialRandomSearch):

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
                                       fit_kwargs.get(
                                           'batch_size',
                                           const.Constant.BATCH_SIZE))
        data_flow = datagen.flow(fit_kwargs['x'],
                                 fit_kwargs['y'],
                                 fit_kwargs['batch_size'],
                                 shuffle=True)
        temp_fit_kwargs = fit_kwargs.copy()
        temp_fit_kwargs.pop('x', None)
        temp_fit_kwargs.pop('y', None)
        temp_fit_kwargs.pop('batch_size', None)

        # Train model
        history = model.fit_generator(data_flow, **temp_fit_kwargs)

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

    def build(self, hp):
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
