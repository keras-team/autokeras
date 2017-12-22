import os
import pickle
from random import randint, random

from keras.layers import Dense, Dropout, MaxPooling1D, MaxPooling2D, MaxPooling3D, Flatten
from keras.losses import categorical_crossentropy
from keras.models import Sequential, load_model
from keras.optimizers import Adam, Adadelta

from autokeras import constant
from autokeras.utils import get_conv_layer_func, extract_config
from autokeras.net_transformer import transform
from autokeras.utils import ModelTrainer


class RandomConvClassifierGenerator:
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


class HillClimbingSearcher:
    def __init__(self, n_classes, input_shape, path, verbose):
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.verbose = verbose
        self.history_configs = []
        self.history = []
        self.path = path
        self.model_count = 0

    def _remove_duplicate(self, models):
        """
        Remove the duplicate in the history_models
        :param models:
        :return:
        """
        ans = []
        for model_a in models:
            model_a_config = extract_config(model_a)
            if model_a_config not in self.history_configs:
                ans.append(model_a)
        return ans

    def generate(self, x_train, y_train, x_test, y_test):
        if not self.history:
            # First model is randomly generated.
            model = RandomConvClassifierGenerator(self.n_classes, self.input_shape).generate()
            self.add_model(model, x_train, y_train, x_test, y_test)

        optimal_accuracy = 0.0
        while self.model_count < constant.MAX_MODEL_NUM:
            model = self.load_best_model()
            new_models = self._remove_duplicate(transform(model))
            for model in new_models:
                self.history_configs.append(extract_config(model))

            for model in new_models:
                if self.model_count < constant.MAX_MODEL_NUM:
                    self.add_model(model, x_train, y_train, x_test, y_test)

            max_accuracy = max(self.history, key=lambda x: x['accuracy'])['accuracy']
            if max_accuracy <= optimal_accuracy:
                break
            optimal_accuracy = max_accuracy

        return self.load_best_model()

    def add_model(self, model, x_train, y_train, x_test, y_test):
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adadelta(),
                      metrics=['accuracy'])
        model.summary()
        ModelTrainer(model, x_train, y_train, x_test, y_test, self.verbose).train_model()
        loss, accuracy = model.evaluate(x_test, y_test)
        model.save(os.path.join(self.path, str(self.model_count) + '.h5'))
        self.history.append({'model_id': self.model_count, 'loss': loss, 'accuracy': accuracy})
        self.history_configs.append(extract_config(model))
        self.model_count += 1
        pickle.dump(self, open(os.path.join(self.path, 'searcher'), 'wb'))

    def load_best_model(self):
        model_id = max(self.history, key=lambda x: x['accuracy'])['model_id']
        return load_model(os.path.join(self.path, str(model_id) + '.h5'))
