import signal
import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.models import Sequential
from sklearn.gaussian_process import GaussianProcessRegressor

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


def my_layer():
    """test one specify layer"""
    a = Input(shape=(3, 3, 2))
    b = WeightedAdd()(a)
    model = Model(inputs=a, outputs=b)
    data = np.ones((1, 3, 3, 2))
    print(model.predict_on_batch(data))
    model.compile(optimizer='Adam', loss=mean_squared_error)
    model.fit(data, data, epochs=1000)
    print(model.predict_on_batch(data))


def gpr():
    gpr = GaussianProcessRegressor()
    gpr.fit([[0, 1, 0, 1]], [1])
    print(gpr.predict([[1, 0, 1, 0]]))
    print(gpr.predict([[0, 1, 0, 1]]))
    # Conclusion: GPR can work with single fit.


def long_function_call():
    a = 1
    for i in range(int(1e10)):
        a += 1


def time_limit():

    def signal_handler(signum, frame):
        raise Exception("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(3)  # Ten seconds
    try:
        long_function_call()
    except Exception as msg:
        print(msg)
        print("Timed is up!")

time_limit()
