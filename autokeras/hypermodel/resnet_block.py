from autokeras.hypermodel.hypermodel_network import ConnectedHyperModel


class ResNetBlock(ConnectedHyperModel):
    def build(self, hp, inputs=None):
        pass


class DenseNetBlock(ConnectedHyperModel):
    def build(self, hp):
        pass


class MlpBlock(ConnectedHyperModel):
    def build(self, hp):
        pass


class AlexNetBlock(ConnectedHyperModel):
    def build(self, hp):
        pass


class CnnBlock(ConnectedHyperModel):
    def build(self, hp):
        pass


class RnnBlock(ConnectedHyperModel):
    def build(self, hp):
        pass


class LstmBlock(ConnectedHyperModel):
    def build(self, hp):
        pass


class SeqToSeqBlock(ConnectedHyperModel):
    def build(self, hp):
        pass


class ImageBlock(ConnectedHyperModel):
    def build(self, hp):
        pass


class NlpBlock(ConnectedHyperModel):
    def build(self, hp):
        pass


class Merge(ConnectedHyperModel):
    def build(self, hp):
        pass


class XceptionBlock(ConnectedHyperModel):
    def build(self, hp):
        pass


class ClassificationHead(ConnectedHyperModel):
    def build(self, hp):
        pass


class RegressionHead(ConnectedHyperModel):
    def build(self, hp):
        pass


class TensorRegressionHead(ConnectedHyperModel):
    def build(self, hp):
        pass


class TensorClassificationHead(ConnectedHyperModel):
    def build(self, hp):
        pass


class ImageInput(ConnectedHyperModel):
    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape

    def build(self, hp):
        pass


class Input(ConnectedHyperModel):
    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape

    def build(self, hp):
        pass

