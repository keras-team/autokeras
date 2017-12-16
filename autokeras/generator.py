from random import randint, random

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


class RandomConvClassifierGenerator(ClassifierGenerator):
    def __init__(self, n_classes, input_shape):
        super().__init__(n_classes, input_shape)
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')

    def _get_pool_layer_func(self):
        pool_funcs = [MaxPooling1D, MaxPooling2D, MaxPooling3D]
        return pool_funcs[len(self.input_shape) - 2]

    def _get_shape(self, dim_size):
        temp_list = [(dim_size,), (dim_size, dim_size), (dim_size, dim_size, dim_size)]
        return temp_list[len(self.input_shape) - 2]

    def generate(self):
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
