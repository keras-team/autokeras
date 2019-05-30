import tensorflow as tf


class HyperNode(object):
    def __init__(self, shape=None):
        super().__init__()
        self.in_hypermodels = []
        self.out_hypermodels = []
        self.shape = shape

    def add_in_hypermodel(self, hypermodel):
        self.in_hypermodels.append(hypermodel)

    def add_out_hypermodel(self, hypermodel):
        self.out_hypermodels.append(hypermodel)

    def build(self, hp):
        raise NotImplementedError


class ImageInput(HyperNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp):
        pass


class Input(HyperNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp):
        return tf.keras.Input(shape=self.shape)
