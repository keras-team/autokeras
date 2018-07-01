from random import randint, random

from keras import Input, Model
from keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D, Dropout, Flatten, Dense, BatchNormalization, \
    Activation, GlobalAveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta, Adam

from autokeras import constant
from autokeras.graph import Graph
from autokeras.layers import get_conv_layer_func, get_ave_layer_func, StubBatchNormalization, StubActivation, StubConv, \
    StubDropout, StubPooling, StubGlobalPooling, StubDense, StubInput
from autokeras.stub import StubModel


class ClassifierGenerator:
    def __init__(self, n_classes, input_shape):
        self.n_classes = n_classes
        self.input_shape = input_shape
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


class DefaultClassifierGenerator(ClassifierGenerator):
    def __init__(self, n_classes, input_shape):
        super().__init__(n_classes, input_shape)

    def generate(self, model_len=constant.MODEL_LEN, model_width=constant.MODEL_WIDTH):
        pool = self._get_pool_layer_func()
        conv = get_conv_layer_func(len(self._get_shape(3)))
        ave = get_ave_layer_func(len(self._get_shape(3)))

        pooling_len = int(model_len / 4)
        model = StubModel()
        model.input_shape = self.input_shape
        model.inputs = [0]
        model.layers.append(StubInput())
        for i in range(model_len):
            model.layers += [StubActivation('relu'),
                             StubConv(model_width, kernel_size=3, func=conv),
                             StubBatchNormalization(),
                             StubDropout(constant.CONV_DROPOUT_RATE)]
            if pooling_len == 0 or ((i + 1) % pooling_len == 0 and i != model_len - 1):
                model.layers.append(StubPooling(func=pool))

        model.layers.append(StubGlobalPooling(ave))
        model.layers.append(StubDense(self.n_classes, activation='softmax'))
        model.outputs = [len(model.layers)]
        for index, layer in enumerate(model.layers):
            layer.input = index
            layer.output = index + 1
        return Graph(model, False)


class RandomConvClassifierGenerator(ClassifierGenerator):
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
