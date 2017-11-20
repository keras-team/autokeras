import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam


class AutoKerasClassifier:
    def __init__(self, verbose=False):
        self.y_encoder = OneHotEncoder()
        self.model = None
        self.train_dim = 0
        self.n_classes = 0
        self.verbose = verbose
        self.n_epochs = 10000

    def fit(self, x_train, y_train):
        x_train = np.array(x_train)
        y_train = np.array(y_train).flatten()

        try:
            x_train = x_train.astype('float64')
        except ValueError:
            raise ValueError('x_train should only contain numerical data.')

        if len(x_train.shape) < 2:
            raise ValueError('x_train should be a 2d array.')

        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError('x_train and y_train should have the same number of instances.')

        self.y_encoder.fit(y_train)
        encoded_y_train = self.y_encoder.transform(y_train)

        self.train_dim = x_train.shape[1]
        self.n_classes = len(set(y_train.flatten()))

        self._build_model()

        if self.verbose:
            self.summary()

        self.model.fit(x_train, encoded_y_train,
                       batch_size=x_train.shape[0],
                       epochs=self.n_epochs,
                       verbose=self.verbose)

    def predict(self, x_test):
        return self.y_encoder.inverse_transform(self.model.predict(x_test, verbose=self.verbose))

    def _build_model(self):
        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=(self.train_dim,)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.n_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(),
                           metrics=['accuracy'])

    def summary(self):
        self.model.summary()


class OneHotEncoder:
    def __init__(self):
        self.data = None
        self.n_classes = 0
        self.labels = None
        self.label_to_vec = {}
        self.int_to_label = {}

    def fit(self, data):
        data = np.array(data).flatten()
        self.labels = set(data)
        self.n_classes = len(self.labels)
        for index, label in enumerate(self.labels):
            vec = np.array([0] * self.n_classes)
            vec[index] = 1
            self.label_to_vec[label] = vec
            self.int_to_label[index] = label

    def transform(self, data):
        data = np.array(data)
        return np.array(list(map(lambda x: self.label_to_vec[x], data)))

    def inverse_transform(self, data):
        return np.array(list(map(lambda x: self.int_to_label[x], np.argmax(np.array(data), axis=1))))
