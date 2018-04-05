import os
import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, MaxPooling2D, Concatenate, Dropout

from autokeras import constant
from autokeras.layers import WeightedAdd


def get_concat_skip_model():
    input_tensor = Input(shape=(5, 5, 3))
    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(input_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    add_input = output_tensor
    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    output_tensor = Concatenate()([output_tensor, add_input])
    add_input = output_tensor
    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    output_tensor = Concatenate()([output_tensor, add_input])
    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(5, activation='relu')(output_tensor)
    output_tensor = Dropout(constant.DENSE_DROPOUT_RATE)(output_tensor)
    output_tensor = Dense(5, activation='softmax')(output_tensor)
    return Model(inputs=input_tensor, outputs=output_tensor)


def get_add_skip_model():
    input_tensor = Input(shape=(5, 5, 3))
    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(input_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    add_input = output_tensor
    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    output_tensor = WeightedAdd()([output_tensor, add_input])
    add_input = output_tensor
    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    output_tensor = WeightedAdd()([output_tensor, add_input])
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(5, activation='relu')(output_tensor)
    output_tensor = Dropout(constant.DENSE_DROPOUT_RATE)(output_tensor)
    output_tensor = Dense(5, activation='softmax')(output_tensor)
    return Model(inputs=input_tensor, outputs=output_tensor)


def get_conv_model():
    input_tensor = Input(shape=(5, 5, 3))
    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(input_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)
    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)
    return Model(inputs=input_tensor, outputs=output_tensor)


def get_conv_data():
    return np.random.rand(1, 5, 5, 3)


def get_conv_dense_model():
    input_tensor = Input(shape=(5, 5, 3))
    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(input_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(5, activation='relu')(output_tensor)
    output_tensor = Dropout(constant.DENSE_DROPOUT_RATE)(output_tensor)
    output_tensor = Dense(5, activation='softmax')(output_tensor)
    return Model(inputs=input_tensor, outputs=output_tensor)


def get_pooling_model():
    input_tensor = Input(shape=(5, 5, 3))
    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(input_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    output_tensor = MaxPooling2D(padding='same')(output_tensor)

    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    output_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation='linear')(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)

    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(5, activation='relu')(output_tensor)
    output_tensor = Dropout(constant.DENSE_DROPOUT_RATE)(output_tensor)
    output_tensor = Dense(5, activation='softmax')(output_tensor)
    return Model(inputs=input_tensor, outputs=output_tensor)


def clean_dir(path):
    for f in os.listdir(path):
        if f != '.gitkeep':
            os.remove(os.path.join(path, f))
