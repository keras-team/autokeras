import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, BatchNormalization, Activation


def get_conv_model():
    input_tensor = Input(shape=(5, 5, 3))
    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(input_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    return Model(inputs=input_tensor, outputs=output_tensor)


def get_conv_data():
    return np.random.rand(1, 5, 5, 3)

