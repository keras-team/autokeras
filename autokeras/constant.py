from keras.layers import Conv3D, Conv2D, Conv1D, Dense

CONV_FUNC_LIST = [Conv1D, Conv2D, Conv3D]
WEIGHTED_LAYER_FUNC_LIST = CONV_FUNC_LIST + [Dense]
MAX_MODEL_NUM = 100
MAX_ITER_NUM = 100000
MIN_LOSS_DEC = 1e-4
MAX_NO_IMPROVEMENT_NUM = 10
LAYER_ATTR = {'Dense': ['units', 'activation'],
              'Dropout': ['rate'],
              'MaxPooling1D': ['pool_size'],
              'MaxPooling2D': ['pool_size'],
              'MaxPooling3D': ['pool_size'],
              'Conv1D': ['filters', 'activation', 'kernel_size'],
              'Conv2D': ['filters', 'activation', 'kernel_size'],
              'Conv3D': ['filters', 'activation', 'kernel_size'],
              'Flatten': []
              }
