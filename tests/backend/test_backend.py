import warnings

from autokeras.constant import Constant
from autokeras.nn.graph import *
from autokeras.backend.tensorflow import *
from tests.common import get_conv_data, get_add_skip_model, get_conv_dense_model, get_pooling_model, \
    get_concat_skip_model

try:
    Constant.BACKEND == 'tensorflow'
    from autokeras.backend import Backend as Backend_TF
except ImportError:
    Backend_TF = None
    warnings.warn('Could not import the TensorFlow backend.')


# def test_produce_keras_model():
#     for graph in [get_conv_dense_model(),
#                   get_add_skip_model(),
#                   get_pooling_model(),
#                   get_concat_skip_model()]:
#         model = graph.produce_keras_model()
#         assert isinstance(model, models.Model)
#
#
# def test_keras_model():
#     for graph in [get_conv_dense_model(),
#                   get_add_skip_model(),
#                   get_pooling_model(),
#                   get_concat_skip_model()]:
#         keras_model = model.KerasModel(graph)
#         keras_model.set_weight_to_graph()
#         assert isinstance(keras_model, model.KerasModel)