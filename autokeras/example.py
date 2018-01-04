import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.models import Sequential

from autokeras.layers import WeightedAdd


def graph_model():
    """test an graph model"""
    a = Input(shape=(32,))
    original_input = a
    b = Dense(32)(a)
    b = Dense(32)(b)
    dense_model1 = Model(inputs=a, outputs=b)
    a = Input(shape=(32,))
    b = Dense(32)(a)
    b = Dense(32)(b)
    dense_model2 = Model(inputs=a, outputs=b)
    dense_model = Sequential([dense_model1, dense_model2])
    print(dense_model1.input_shape)
    print(dense_model1.output_shape)
    print(dense_model.input_shape)
    print(dense_model.output_shape)
    print(dense_model.layers)

    a = dense_model1.output
    b = Dense(32)(a)
    b = Dense(32)(b)
    final_model = Model(inputs=original_input, outputs=b)
    print(final_model.layers)


def test_my_layer():
    """test one specify layer"""
    a = Input(shape=(3, 3, 2))
    b = WeightedAdd()(a)
    model = Model(inputs=a, outputs=b)
    data = np.ones((1, 3, 3, 2))
    print(model.predict_on_batch(data))
    model.compile(optimizer='Adam', loss=mean_squared_error)
    model.fit(data, data, epochs=1000)
    print(model.predict_on_batch(data))


# graph_model()

test_my_layer()
