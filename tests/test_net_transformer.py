from autokeras.net_transformer import *

def test_net_transformer():
    model_a = Sequential()
    model_a.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model_a.add(Conv2D(32, (3, 3), activation='relu'))
    model_a.add(MaxPooling2D(pool_size=(2, 2)))
    model_a.add(Dropout(0.25))
    net_transfromer(model_a)