import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network.multilayer_perceptron import BaseMultilayerPerceptron

from autokeras.generator import ClassifierGenerator
from autokeras.preprocessor import OneHotEncoder

MAX_MODEL_NUM = 100
MAX_ITER_NUM = 100000
MIN_LOSS_DEC = 1e-4
MAX_NO_IMPROVEMENT_NUM = 10


class ClassifierBase:
    def __init__(self, verbose=False):
        self.y_encoder = OneHotEncoder()
        self.model = None
        self.verbose = verbose
        self.generator = None
        self.history = []
        self.training_losses = []

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
        self.generator = self._get_generator(n_classes, input_shape)

        # Transform y_train.
        self.y_encoder.fit(y_train)
        y_train = self.y_encoder.transform(y_train)

        # Divide training data into training and testing data.
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

        for i in range(MAX_MODEL_NUM):
            model = self.generator.generate()

            if self.verbose:
                model.summary()

            self._train_model(model, x_train, y_train, x_test, y_test)
            # The same auto batch size_train, y_train strategy as sklearn
            loss, accuracy = self.model.evaluate(x_test, y_test)
            self.history.append({'model': self.model, 'loss': loss, 'accuracy': accuracy})
        self.history.sort(key=lambda x: x['accuracy'])
        self.model = self.history[-1]['model']

    def predict(self, x_test):
        return self.y_encoder.inverse_transform(self.model.predict(x_test, verbose=self.verbose))

    def summary(self):
        self.model.summary()

    def _get_generator(self, n_classes, input_shape):
        return None

    def _converged(self, loss):
        self.training_losses.append(loss)

        if loss > (self.minimum_loss - MIN_LOSS_DEC):
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0

        if loss < self.minimum_loss:
            self.minimum_loss = loss

        return self._no_improvement_count > MAX_NO_IMPROVEMENT_NUM

    def _train_model(self, model, x_train, y_train, x_test, y_test):
        self.training_losses = []
        self.minimum_loss = float('inf')
        for _ in range(MAX_ITER_NUM):
            model.fit(x_train, y_train,
                      batch_size=min(x_train.shape[0], 200),
                      verbose=self.verbose)
            loss, _ = model.evaluate(x_test, y_test)
            if self._converged(loss):
                break
        pass


class Classifier(ClassifierBase):
    def __init__(self):
        super().__init__()

    def _validate(self, x_train, y_train):
        super()._validate(x_train, y_train)

    def _get_generator(self, n_classes, input_shape):
        return ClassifierGenerator(n_classes, input_shape)


class ImageClassifier(ClassifierBase):
    def __init__(self):
        super().__init__()
