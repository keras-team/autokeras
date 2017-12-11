from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam


class ClassifierGenerator:
    def __init__(self, n_classes, input_shape):
        self.n_classes = n_classes
        self.input_shape = input_shape

    def generate(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(self.input_shape,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(),
                           metrics=['accuracy'])
        return model
