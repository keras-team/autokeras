import tensorflow as tf


class Node(object):
    def __init__(self, shape=None):
        super().__init__()
        self.in_blocks = []
        self.out_blocks = []
        self.shape = shape

    def add_in_block(self, hypermodel):
        self.in_blocks.append(hypermodel)

    def add_out_block(self, hypermodel):
        self.out_blocks.append(hypermodel)

    def build(self):
        return tf.keras.Input(shape=self.shape)

    def clear_edges(self):
        self.in_blocks = []
        self.out_blocks = []


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
