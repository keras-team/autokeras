from autokeras.generator import *
from autokeras.graph import TorchModel


def test_default_generator():
    generator = DefaultClassifierGenerator(3, (28, 28, 1))
    graph = generator.generate()
    model = graph.produce_model()
    assert isinstance(model, TorchModel)
