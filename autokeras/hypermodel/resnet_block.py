from autokeras import HyperModel


class ResNetBlock(HyperModel):
    def build(self, hp):
        pass


class DenseNetBlock(HyperModel):
    def build(self, hp):
        pass


class MlpBlock(HyperModel):
    def build(self, hp):
        pass


class AlexNetBlock(HyperModel):
    def build(self, hp):
        pass


class CnnBlock(HyperModel):
    def build(self, hp):
        pass


class RnnBlock(HyperModel):
    def build(self, hp):
        pass


class LstmBlock(HyperModel):
    def build(self, hp):
        pass


class SeqToSeqBlock(HyperModel):
    def build(self, hp):
        pass


class ImageBlock(HyperModel):
    def build(self, hp):
        pass


class NlpBlock(HyperModel):
    def build(self, hp):
        pass


class Merge(HyperModel):
    def build(self, hp):
        pass


class XceptionBlock(HyperModel):
    def build(self, hp):
        pass


class ClassificationHead(HyperModel):
    def build(self, hp):
        pass


class RegressionHead(HyperModel):
    def build(self, hp):
        pass


class TensorRegressionHead(HyperModel):
    def build(self, hp):
        pass


class TensorClassificationHead(HyperModel):
    def build(self, hp):
        pass


class ImageInput(HyperModel):
    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape

    def build(self, hp):
        pass


class Input(HyperModel):
    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape

    def build(self, hp):
        pass

