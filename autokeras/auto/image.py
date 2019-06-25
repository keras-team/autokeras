import tensorflow as tf

from autokeras.auto import processor
from autokeras.auto import auto_model
from autokeras.hypermodel import hyper_block
from autokeras.hypermodel import hyper_node
from autokeras.hypermodel import hyper_head


class SupervisedImagePipeline(auto_model.AutoModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_block = hyper_block.ImageBlock()
        self.head = None
        self.normalizer = processor.Normalizer()

    def fit(self, x=None, y=None, **kwargs):
        self.normalizer.fit(x)
        self.inputs = [hyper_node.ImageInput()]
        super().fit(x=self.normalizer.transform(x), y=y, **kwargs)

    def build(self, hp):
        input_node = self.inputs[0].build(hp)
        output_node = self.image_block.build(hp, input_node)
        output_node = self.head.build(hp, output_node)
        model = tf.keras.Model(input_node, output_node)
        optimizer = hp.Choice('optimizer',
                              ['adam',
                               'adadelta',
                               'sgd'])

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
