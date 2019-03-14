from autokeras.backend import torch
from autokeras.constant import Constant


class Backend:
    backend = torch if Constant.BACKEND == 'torch' else None

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
