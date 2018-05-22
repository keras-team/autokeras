from random import randint, random

from keras import Input, Model
from keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D, Dropout, Flatten, Dense, BatchNormalization, \
    Activation, GlobalAveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta, Adam

from autokeras import constant
from autokeras.layers import get_conv_layer_func, get_ave_layer_func


class ClassifierGenerator:
    """The base class of classifier generators.

    ClassifierGenerator is the base class of all classifier generator classes.
    It is used for generating classifier models.

    Attributes:
        n_classes: Number of classes in the input data.
        input_shape: A tuple of integers containing the size of each dimension of the input data,
            excluding the dimension of number of training examples. The length of the tuple should
            between two and four inclusively.
    """

    def __init__(self, n_classes, input_shape):
        self.n_classes = n_classes
        self.input_shape = input_shape
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')

    def _get_pool_layer_func(self):
        """Return MaxPooling function based on the dimension of input shape."""
        pool_funcs = [MaxPooling1D, MaxPooling2D, MaxPooling3D]
        return pool_funcs[len(self.input_shape) - 2]

    def _get_shape(self, dim_size):
        """Return filter shape tuple based on the dimension of input shape."""
        temp_list = [(dim_size,), (dim_size, dim_size), (dim_size, dim_size, dim_size)]
        return temp_list[len(self.input_shape) - 2]


class DefaultClassifierGenerator(ClassifierGenerator):
    """A classifier generator always generates models with the same default architecture and configuration."""

    def __init__(self, n_classes, input_shape):
        super().__init__(n_classes, input_shape)

    def generate(self, model_len=constant.MODEL_LEN, model_width=constant.MODEL_WIDTH):
        """Return the default classifier model that has been compiled."""
        pool = self._get_pool_layer_func()
        conv = get_conv_layer_func(len(self._get_shape(3)))
        ave = get_ave_layer_func(len(self._get_shape(3)))

        pooling_len = int(model_len / 4)
        output_tensor = input_tensor = Input(shape=self.input_shape)
        for i in range(model_len):
            output_tensor = BatchNormalization()(output_tensor)
            output_tensor = Activation('relu')(output_tensor)
            output_tensor = conv(model_width, kernel_size=self._get_shape(3), padding='same')(output_tensor)
            output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)
            if (i + 1) % pooling_len == 0 and i != model_len - 1:
                output_tensor = pool(padding='same')(output_tensor)

        output_tensor = ave()(output_tensor)
        output_tensor = Dense(self.n_classes, activation='softmax')(output_tensor)
        return Model(inputs=input_tensor, outputs=output_tensor)


class RandomConvClassifierGenerator(ClassifierGenerator):
    """A classifier generator that generates random convolutional neural networks."""

    def __init__(self, n_classes, input_shape):
        super().__init__(n_classes, input_shape)

    def generate(self):
        """Return the random generated CNN model."""
        conv_num = randint(1, 10)
        dense_num = randint(1, 10)
        dropout_rate = random()
        filter_size = randint(1, 2) * 2 + 1
        pool_size = randint(2, 3)
        filter_shape = self._get_shape(filter_size)
        pool_shape = self._get_shape(pool_size)
        pool = self._get_pool_layer_func()
        conv = get_conv_layer_func(len(filter_shape))

        input_tensor = Input(shape=self.input_shape)
        output_tensor = input_tensor
        for i in range(conv_num):
            kernel_num = randint(10, 30)
            output_tensor = conv(kernel_num, filter_shape,
                                 padding='same')(output_tensor)
            output_tensor = BatchNormalization()(output_tensor)
            output_tensor = Activation('relu')(output_tensor)
            if random() > 0.5:
                output_tensor = pool(pool_size=pool_shape, padding='same')(output_tensor)
            if random() > 0.5:
                output_tensor = Dropout(dropout_rate)(output_tensor)
        output_tensor = Flatten()(output_tensor)
        for i in range(dense_num):
            node_num = randint(128, 1024)
            output_tensor = Dense(node_num, activation='relu')(output_tensor)
            if random() > 0.5:
                output_tensor = Dropout(dropout_rate)(output_tensor)
        output_tensor = Dense(self.n_classes, activation='softmax')(output_tensor)
        model = Model(input_tensor, output_tensor)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model
