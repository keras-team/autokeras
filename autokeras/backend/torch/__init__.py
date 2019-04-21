import numpy as np
from functools import reduce
from autokeras.backend.torch.model import produce_model
from autokeras.backend.torch.data_transformer import ImageDataTransformer, DataTransformerMlp
from autokeras.backend.torch.model_trainer import ModelTrainer, get_device
from autokeras.backend.torch.loss_function import *
from autokeras.backend.torch.metric import *


def predict(torch_model, loader):
    outputs = []
    with torch.no_grad():
        for index, inputs in enumerate(loader):
            outputs.append(torch_model(inputs).numpy())
    output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
    return output


