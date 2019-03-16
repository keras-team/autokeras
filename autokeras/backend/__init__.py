from autokeras.backend import torch, tensorflow
from autokeras.constant import Constant


class Backend:
    backend = torch if Constant.BACKEND == 'torch' else tensorflow

    def __init__(self):
        pass

    @classmethod
    def get_image_transformer(cls, x_train, augment=None):
        return cls.backend.ImageDataTransformer(x_train, augment=augment)

    @classmethod
    def produce_model(cls, graph):
        return cls.backend.produce_model(graph)

    @classmethod
    def get_model_trainer(cls, **kwargs):
        return cls.backend.ModelTrainer(**kwargs)

    @classmethod
    def classification_loss(cls, prediction, target):
        return cls.backend.classification_loss(prediction, target)

    @classmethod
    def regression_loss(cls, prediction, target):
        return cls.backend.regression_loss(prediction, target)

    @classmethod
    def binary_classification_loss(cls, prediction, target):
        return cls.backend.binary_classification_loss(prediction, target)

    @classmethod
    def classification_metric(cls, prediction, target):
        return cls.backend.classification_metric(prediction, target)

    @classmethod
    def regression_metric(cls, prediction, target):
        return cls.backend.regression_metric(prediction, target)

    @classmethod
    def binary_classification_metric(cls, prediction, target):
        return cls.backend.binary_classification_metric(prediction, target)

    @classmethod
    def predict(cls, model, loader):
        return cls.backend.predict(model, loader)

    @classmethod
    def get_device(cls):
        return cls.backend.get_device()
