from random import randint, random

from keras import Input
from keras.layers import Dense, Dropout, MaxPooling1D, MaxPooling2D, MaxPooling3D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from autokeras.utils import get_conv_layer_func


class ClassifierGenerator:
    def __init__(self, n_classes, input_shape):
        self.n_classes = n_classes
        self.input_shape = input_shape

    def generate(self):
        pass


class RandomCovClassifierGenerator(ClassifierGenerator):
    def __init__(self, n_classes, input_shape):
        super().__init__(n_classes, input_shape)

    def _get_pool_layer_func(self):
        if len(self.input_shape) == 1:
            return MaxPooling1D
        elif len(self.input_shape) == 2:
            return MaxPooling2D
        elif len(self.input_shape) == 3:
            return MaxPooling3D
        raise ValueError('The input dimension is too high.')

    def _get_shape(self, dim_size):
        if len(self.input_shape) == 1:
            return dim_size,
        elif len(self.input_shape) == 2:
            return dim_size, dim_size
        elif len(self.input_shape) == 3:
            return dim_size, dim_size, dim_size
        raise ValueError('The input dimension is too high.')

    def generate(self):
        conv_num = randint(1, 10)
        dense_num = randint(1, 10)
        dropout_rate = random()
        filter_size = randint(1, 2) * 2 + 1
        pool_size = randint(2, 3)
        conv = get_conv_layer_func(len(self.input_shape))
        filter_shape = self._get_shape(filter_size)
        pool_shape = self._get_shape(pool_size)
        pool = self._get_pool_layer_func()

        model = Sequential()
        model.add(Input(shape=self.input_shape))
        for i in range(conv_num):
            kernel_num = randint(10, 30)
            model.add(conv(kernel_num, kernel_size=filter_shape, activation='relu'))
            if random() > 0.5:
                model.add(pool(pool_size=pool_shape))
            if random() > 0.5:
                model.add(Dropout(dropout_rate))
        model.add(Flatten())
        for i in range(dense_num):
            node_num = random(128, 1024)
            model.add(Dense(node_num, activation='relu'))
            if random() > 0.5:
                model.add(Dropout(dropout_rate))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model
