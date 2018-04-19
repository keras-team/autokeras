from keras.layers import Conv3D, Conv2D, Conv1D, Dense

from autokeras.layers import StubDense, StubConv

CONV_FUNC_LIST = [Conv1D, Conv2D, Conv3D, StubConv]
WEIGHTED_LAYER_FUNC_LIST = CONV_FUNC_LIST + [Dense, StubDense]
DEFAULT_SAVE_PATH = '/tmp/autokeras/'


# Searcher

MAX_MODEL_NUM = 1000
ACQ_EXPLOITATION_DEPTH = 4

# Model Defaults

DENSE_DROPOUT_RATE = 0.5
CONV_DROPOUT_RATE = 0.25
CONV_BLOCK_SIZE = 4
DENSE_BLOCK_SIZE = 2

# ModelTrainer

N_NEIGHBORS = 8
DATA_AUGMENTATION = True
MAX_ITER_NUM = 200
MIN_LOSS_DEC = 1e-4
MAX_NO_IMPROVEMENT_NUM = 100
EPOCHS_EACH = 1
MAX_BATCH_SIZE = 32
LIMIT_MEMORY = False
