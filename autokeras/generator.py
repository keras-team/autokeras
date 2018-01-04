from random import randint, random

from keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D, Dropout, Flatten, Dense
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta, Adam

from autokeras.utils import get_conv_layer_func


class ClassifierGenerator:
    """Base class of ClassifierGenerator

    ClassifierGenerator is the base class of all ClassifierGenerator classes, classifier_generator is used
    to generate model

    Attributes:
        n_classes: number of class in the input data
        input_shape: Arbitrary, although all dimensions in the input shaped must be fixed.
                   Use the keyword argument input_shape (tuple of integers, does not include the batch axis)
                   when using this layer as the first layer in a model.
    """
    def __init__(self, n_classes, input_shape):
        """Inits ClassifierBase with n_classes, input_shape"""
        self.n_classes = n_classes
        self.input_shape = input_shape
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')

    def _get_pool_layer_func(self):
        """Return MaxPooling function based on the dimension of input shape"""
        pool_funcs = [MaxPooling1D, MaxPooling2D, MaxPooling3D]
        return pool_funcs[len(self.input_shape) - 2]

    def _get_shape(self, dim_size):
        """Return shape tuple based on the dimension of input shape"""
        temp_list = [(dim_size,), (dim_size, dim_size), (dim_size, dim_size, dim_size)]
        return temp_list[len(self.input_shape) - 2]


class DefaultClassifierGenerator(ClassifierGenerator):
    """Default classifierGenerator class inherited from ClassifierGenerator class"""
    def __init__(self, n_classes, input_shape):
        """Inits DefaultClassifierGenerator with n_classes, input_shape"""
        super().__init__(n_classes, input_shape)

    def generate(self):
        """return one Sequential model that has been compiled"""
        model = Sequential()
        pool = self._get_pool_layer_func()
        conv = get_conv_layer_func(len(self._get_shape(3)))
        model.add(conv(32, kernel_size=self._get_shape(3),
                       activation='relu',
                       padding='same',
                       input_shape=self.input_shape))
        model.add(conv(64, self._get_shape(3),
                       padding='same',
                       activation='relu'))
        model.add(pool(pool_size=self._get_shape(2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adadelta(),
                      metrics=['accuracy'])
        return model


class RandomConvClassifierGenerator(ClassifierGenerator):
    """Random Convolution ClassifierGenerator based on the ClassifierGenerator"""
    def __init__(self, n_classes, input_shape):
        """Inits RandomConvClassifierGenerator with n_classes, input_shape"""
        super().__init__(n_classes, input_shape)

    def generate(self):
        """return one Sequential model that has been compiled"""
        conv_num = randint(1, 10)
        dense_num = randint(1, 10)
        dropout_rate = random()
        filter_size = randint(1, 2) * 2 + 1
        pool_size = randint(2, 3)
        filter_shape = self._get_shape(filter_size)
        pool_shape = self._get_shape(pool_size)
        pool = self._get_pool_layer_func()
        conv = get_conv_layer_func(len(filter_shape))

        model = Sequential()
        for i in range(conv_num):
            kernel_num = randint(10, 30)
            if i == 0:
                model.add(conv(kernel_num,
                               input_shape=self.input_shape,
                               kernel_size=filter_shape,
                               activation='relu',
                               padding='same'))
            else:
                model.add(conv(kernel_num,
                               kernel_size=filter_shape,
                               activation='relu',
                               padding='same'))
            if random() > 0.5:
                model.add(pool(pool_size=pool_shape, padding='same'))
            if random() > 0.5:
                model.add(Dropout(dropout_rate))
        model.add(Flatten())
        for i in range(dense_num):
            node_num = randint(128, 1024)
            model.add(Dense(node_num, activation='relu'))
            if random() > 0.5:
                model.add(Dropout(dropout_rate))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model
