from autokeras.nn.generator import *
from autokeras.nn.graph import TorchModel


def test_default_generator():
    generator = CnnGenerator(3, (28, 28, 1))
    graph = generator.generate()
    model = graph.produce_model()
    assert isinstance(model, TorchModel)
