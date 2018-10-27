from autokeras.nn.layers import *
import numpy as np


def test_global_layer():
    layer = GlobalAvgPool2d()
    inputs = torch.Tensor(np.ones((100, 50, 30, 40)))
    assert layer(inputs).size() == (100, 50)
