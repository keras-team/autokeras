from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten

from autokeras.net_combinator import *


# def test_combine():
#     model1 = Sequential([Conv2D(10, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
#                          MaxPooling2D(pool_size=(2, 2)),
#                          Dropout(0.25),
#                          Flatten(),
#                          Dense(100, activation='relu'),
#                          Dense(100, activation='relu'),
#                          Dropout(0.5),
#                          Dense(10, activation='softmax')])
#     model2 = Sequential([Conv2D(10, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
#                          Conv2D(30, kernel_size=(3, 3), activation='relu', padding='same'),
#                          MaxPooling2D(pool_size=(2, 2)),
#                          Dropout(0.25),
#                          Flatten(),
#                          Dense(100, activation='relu'),
#                          Dropout(0.5),
#                          Dense(10, activation='softmax')])
#     new_model = combine(model1, model2)
#     assert len(new_model.layers) == 9
