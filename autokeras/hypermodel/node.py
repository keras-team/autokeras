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

    def build(self):
        return tf.keras.Input(shape=self.shape)

    def clear_edges(self):
        self.in_hypermodels = []
        self.out_hypermodels = []


class TextNode(Node):
    pass


class Input(Node):
    pass


class ImageInput(Input):
    pass


class TextInput(Input, TextNode):
    pass


class StructuredInput(Input):
    pass


class TimeSeriesInput(Input):
    pass
