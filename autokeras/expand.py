import numpy as np
from keras import Input, optimizers, losses
from keras.datasets import mnist
from keras.engine import Model
from keras.layers import Dense, Add, Concatenate, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils


def mlp():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    input_x = Input(shape=(784,))

    layer_a1 = Dense(30, activation='relu')
    a1 = layer_a1(input_x)

    layer_a2 = Dense(40, activation='relu')
    a2 = layer_a2(a1)

    layer_output_a = Dense(10, activation='softmax')
    output_a = layer_output_a(a2)

    model_a = Model(input_x, output_a)
    model_a.compile(optimizer='Adam', loss='categorical_crossentropy')
    model_a.fit(x_train, y_train, epochs=50, batch_size=256, shuffle=True)
    print(model_a.evaluate(x_test, y_test))

    layer_b1 = Dense(50, activation='relu')
    b1 = layer_b1(input_x)

    layer_b2 = Dense(60, activation='relu')
    b2 = layer_b2(b1)

    layer_output_b = Dense(10, activation='softmax')
    output_b = layer_output_b(b2)

    model_b = Model(input_x, output_b)
    model_b.compile(optimizer='Adam', loss='categorical_crossentropy')
    model_b.fit(x_train, y_train, epochs=50, batch_size=256, shuffle=True)
    print(model_b.evaluate(x_test, y_test))

    layer_ab = Dense(60, activation='relu')
    ab = layer_ab(a1)

    layer_ba = Dense(40, activation='relu')
    ba = layer_ba(b1)

    layer_a3 = Add()
    a3 = layer_a3([a2, ba])

    layer_b3 = Add()
    b3 = layer_b3([b2, ab])

    layer_l3 = Concatenate(axis=-1)
    l3 = layer_l3([a3, b3])
    layer_output = Dense(10, activation='softmax')
    output_x = layer_output(l3)

    layer_a1.trainable = False
    layer_a2.trainable = False
    layer_b1.trainable = False
    layer_b2.trainable = False

    model = Model(input_x, output_x)
    model.compile(optimizer='Adam', loss='categorical_crossentropy')
    history = model.fit(x_train, y_train, epochs=50, batch_size=256, shuffle=True)
    print(model.evaluate(x_test, y_test))

    print(layer_a1.get_weights()[0].shape)
    print(layer_a1.get_weights()[1].shape)
    print(layer_b1.get_weights()[0].shape)
    print(layer_b1.get_weights()[1].shape)

    c1_weights = np.concatenate((layer_a1.get_weights()[0], layer_b1.get_weights()[0]), axis=1)
    c1_bias = np.concatenate((layer_a1.get_weights()[1], layer_b1.get_weights()[1]), axis=0)
    print(c1_weights.shape)
    print(c1_bias.shape)

    c2_weights1 = np.concatenate((layer_a2.get_weights()[0], layer_ab.get_weights()[0]), axis=1)
    c2_weights2 = np.concatenate((layer_ba.get_weights()[0], layer_b2.get_weights()[0]), axis=1)
    c2_weights = np.concatenate((c2_weights1, c2_weights2), axis=0)
    c2_bias1 = np.concatenate((layer_a2.get_weights()[1], layer_ab.get_weights()[1]), axis=0)
    c2_bias2 = np.concatenate((layer_ba.get_weights()[1], layer_b2.get_weights()[1]), axis=0)
    c2_bias = np.mean((c2_bias1, c2_bias2), axis=0)
    print(c2_weights.shape)
    print(c2_bias.shape)

    output_c_weights = layer_output.get_weights()[0]
    output_c_bias = layer_output.get_weights()[1]

    input_c = Input(shape=(784,))
    layer_c1 = Dense(80, activation='relu')
    c1 = layer_c1(input_c)
    layer_c2 = Dense(100, activation='relu')
    c2 = layer_c2(c1)
    layer_output_c = Dense(10, activation='softmax')
    output_c = layer_output_c(c2)

    layer_c1.set_weights([c1_weights, c1_bias])
    layer_c2.set_weights([c2_weights, c2_bias])
    layer_output_c.set_weights([output_c_weights, output_c_bias])

    model_c = Model(input_c, output_c)
    model_c.compile(optimizer='Adam', loss='categorical_crossentropy')
    history = model.fit(x_train, y_train, epochs=50, batch_size=256, shuffle=True)


def get_data():
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def cnn():
    batch_size = 600
    num_classes = 10
    epochs = 1

    x_train, y_train, x_test, y_test = get_data()
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def cnn2():
    x_train, y_train, x_test, y_test = get_data()
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    num_classes = 10

    input1 = Input(shape=input_shape)
    layer_a1 = Conv2D(10, kernel_size=(3, 3), activation='relu')
    a1 = layer_a1(input1)
    layer_a2 = Conv2D(30, kernel_size=(3, 3), activation='relu')
    a2 = layer_a2(a1)
    a2 = MaxPooling2D(pool_size=(2, 2))(a2)
    a2 = Dropout(0.25)(a2)
    a2 = Flatten()(a2)
    layer_a3 = Dense(100, activation='relu')
    a3 = layer_a3(a2)
    a3 = Dropout(0.5)(a3)
    layer_a4 = Dense(num_classes, activation='softmax')
    a4 = layer_a4(a3)
    model_a = Model(input1, a4)
    model_a.compile(loss=losses.categorical_crossentropy,
                    optimizer=optimizers.Adadelta(),
                    metrics=['accuracy'])


if __name__ == '__main__':
    cnn()
