import warnings

from autokeras.constant import Constant

try:
    Constant.BACKEND == 'tensorflow'
    from autokeras.backend import Backend as Backend_TF
except ImportError:
    Backend_TF = None
    warnings.warn('Could not import the TensorFlow backend.')