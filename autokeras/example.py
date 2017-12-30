import tensorflow as tf
from keras import Input
from keras.engine import Model
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.models import Sequential

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class MyLayer(Layer):

    def __init__(self, **kwargs):
        self.weight = K.variable(2.0)
        self.kernel = None
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=input_shape[1],
        #                               initializer='uniform',
        #                               trainable=True)
        self._trainable_weights.append(self.weight)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        return K.tf.scalar_mul(self.weight, x)

    def compute_output_shape(self, input_shape):
        return input_shape


def graph_model():
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
    a = Input(shape=(3, 3, 2))
    b = MyLayer()(a)
    model = Model(inputs=a, outputs=b)
    data = np.ones((1, 3, 3, 2))
    print(model.predict_on_batch(data))
    model.compile(optimizer='Adam', loss=mean_squared_error)
    model.fit(data, data, epochs=1000)
    print(model.predict_on_batch(data))


# graph_model()

test_my_layer()

