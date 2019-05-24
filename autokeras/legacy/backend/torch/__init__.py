from autokeras.legacy.backend.torch.data_transformer import ImageDataTransformer, DataTransformerMlp
from autokeras.legacy.backend.torch.loss_function import *
from autokeras.legacy.backend.torch.metric import *
from autokeras.legacy.backend.torch.model import *
from autokeras.legacy.backend.torch.model_trainer import *


def predict(torch_model, loader):
    outputs = []
    with torch.no_grad():
        for index, inputs in enumerate(loader):
            outputs.append(torch_model(inputs).numpy())
    output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
    return output


