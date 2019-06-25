import tensorflow as tf


class Node(object):
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


class Input(Node):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp):
        return tf.keras.Input(shape=self.shape)


class ImageInput(Node):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp):
        return tf.keras.Input(shape=self.shape)


class TextInput(Node):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp):
        pass


class StructuredInput(Node):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp):
        pass


class TimeSeriesInput(Node):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp):
        pass
