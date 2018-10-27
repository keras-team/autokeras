from autokeras.nn.generator import *
from autokeras.nn.graph import TorchModel
import numpy as np
import torch


def test_default_generator():
    generator = CnnGenerator(3, (28, 28, 1))
    graph = generator.generate()
    model = graph.produce_model()
    inputs = torch.Tensor(np.ones((100, 1, 28, 28)))
    print(model(inputs).size())
    assert isinstance(model, TorchModel)
