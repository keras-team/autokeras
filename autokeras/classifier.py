import numpy as np
import pickle
import os
from keras.models import load_model
from sklearn.model_selection import train_test_split

from autokeras import constant
from autokeras.generator import RandomConvClassifierGenerator
from autokeras.preprocessor import OneHotEncoder
from autokeras.utils import ModelTrainer, clear_path


def load_from_path(path=constant.DEFAULT_SAVE_PATH):
    classifier = pickle.load(os.path.join(path, 'config'))
    return classifier


class ClassifierBase:
    def __init__(self, verbose=False, generator_type=None, path=constant.DEFAULT_SAVE_PATH):
        self.y_encoder = None
        self.model = None
        self.verbose = verbose
        self.generator = None
        self.generator_type = generator_type
        self.history = []
        self.training_losses = []
        self.path = path
        self.current_iteration = 0

    def _validate(self, x_train, y_train):
        try:
            x_train = x_train.astype('float64')
        except ValueError:
            raise ValueError('x_train should only contain numerical data.')

        if len(x_train.shape) < 2:
            raise ValueError('x_train should at least has 2 dimensions.')

        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError('x_train and y_train should have the same number of instances.')

    def fit(self, x_train, y_train):
        x_train = np.array(x_train)
        y_train = np.array(y_train).flatten()

        self._validate(x_train, y_train)

        input_shape = x_train.shape[1:]
        n_classes = len(set(y_train.flatten()))

        if self.generator is None:
            self.generator = self._get_generator(n_classes, input_shape)

        # Transform y_train.
        if self.y_encoder is None:
            self.y_encoder = OneHotEncoder()
            self.y_encoder.fit(y_train)
        y_train = self.y_encoder.transform(y_train)

        # Divide training data into training and testing data.
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

        for i in range(self.current_iteration, constant.MAX_MODEL_NUM):
            model = self.generator.generate()

            if self.verbose:
                model.summary()

            ModelTrainer(model, x_train, y_train, x_test, y_test, self.verbose).train_model()
            model.save(str(i) + '.h5')
            loss, accuracy = model.evaluate(x_test, y_test)
            self.history.append({'model_id': i, 'loss': loss, 'accuracy': accuracy})
            self.current_iteration = i
            pickle.dump(self, os.path.join(self.path, 'config'))

        self.history.sort(key=lambda x: x['accuracy'])
        self.model = load_model(self.history[-1]['model_id'] + '.h5')

    def predict(self, x_test):
        return self.y_encoder.inverse_transform(self.model.predict(x_test, verbose=self.verbose))

    def summary(self):
        self.model.summary()

    def _get_generator(self, n_classes, input_shape):
        return None


class Classifier(ClassifierBase):
    def __init__(self):
        super().__init__()

    def _validate(self, x_train, y_train):
        super()._validate(x_train, y_train)


class ImageClassifier(ClassifierBase):
    def __init__(self, verbose=True, generator_type='random', path=constant.DEFAULT_SAVE_PATH):
        super().__init__(verbose, generator_type, path)

    def _get_generator(self, n_classes, input_shape):
        if self.generator_type == 'random':
            return RandomConvClassifierGenerator(n_classes, input_shape)
        return None
