from keras import Input
from keras.engine import Model
from keras.layers import Dense
from keras.models import Sequential


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

graph_model()
