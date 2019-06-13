import tensorflow as tf

from autokeras.auto import auto_pipeline
from autokeras.auto import processor
from autokeras.hypermodel import hyper_block, hyper_node
from autokeras.hypermodel import hyper_head
from autokeras import tuner


class TextTuner(tuner.SequentialRandomSearch):

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

        # Optional recompile
        if not model.optimizer:
            model.compile()
        elif self.optimizer or self.loss or self.metrics:
            compile_kwargs = {
                'optimizer': model.optimizer,
                'loss': model.loss,
                'metrics': model.metrics,
            }
            if self.loss:
                compile_kwargs['loss'] = self.loss
            if self.optimizer:
                compile_kwargs['optimizer'] = self.optimizer
            if self.metrics:
                compile_kwargs['metrics'] = self.metrics
            model.compile(compile_kwargs['optimizer'], compile_kwargs['loss'], compile_kwargs['metrics'])

        # Train model
        # TODO: reporting presumably done with a callback, record the hp and performances
        history = model.fit(**fit_kwargs)

        metric_name = model.metrics_names[1]
        feedback = history.history['val_' + metric_name][-1]
        return model, feedback


class TextSupervised(auto_pipeline.AutoPipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_block = hyper_block.TextBlock()
        self.head = None

    def fit(self, x=None, y=None, **kwargs):
        self.tuner = TextTuner(self, metrics=self.head.metrics)
        self.inputs = [hyper_node.TextInput()]
        super().fit(x=x, y=y, **kwargs)

    def build(self, hp):
        input_node = self.inputs[0].build(hp)
        output_node = self.text_block.build(hp, input_node)
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


class TextClassifier(TextSupervised):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = hyper_head.ClassificationHead()
        self.y_encoder = processor.OneHotEncoder()

    def fit(self, x=None, y=None, **kwargs):
        self.y_encoder.fit(y)
        self.head.output_shape = (self.y_encoder.n_classes,)
        super().fit(x=x, y=self.y_encoder.transform(y), **kwargs)

    def predict(self, x, **kwargs):
        return self.y_encoder.inverse_transform(super().predict(x, **kwargs))


class TextRegressor(TextSupervised):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = hyper_head.RegressionHead()

    def fit(self, x=None, y=None, **kwargs):
        self.head.output_shape = (1,)
        super().fit(x=x, y=y, **kwargs)

    def predict(self, x, **kwargs):
        return super().predict(x, **kwargs).flatten()
