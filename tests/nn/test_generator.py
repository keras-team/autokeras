from autokeras.nn.generator import *
from autokeras.nn.graph import TorchModel


def test_default_cnn_generator():
    generator = CnnGenerator(3, (28, 28, 1))
    graph = generator.generate()
    model = graph.produce_model()
    assert isinstance(model, TorchModel)


def test_default_mlp_generator():
    generator = MlpGenerator(5, (4,))
    graph = generator.generate(3, [9, 8, 6])
    model = graph.produce_model()
    assert isinstance(model, TorchModel)