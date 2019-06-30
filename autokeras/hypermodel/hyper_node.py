import tensorflow as tf
from autokeras.hypermodel import hyper_block


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


class Input(Node):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def related_block():
        return hyper_block.GeneralBlock()


class ImageInput(Input):

    @staticmethod
    def related_block():
        return hyper_block.ImageBlock()


class TextInput(Input):

    @staticmethod
    def related_block():
        return hyper_block.TextBlock()


class StructuredInput(Input):

    @staticmethod
    def related_block():
        return hyper_block.StructuredBlock()


class TimeSeriesInput(Input):

    @staticmethod
    def related_block():
        return hyper_block.TimeSeriesBlock()
