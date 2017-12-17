from autokeras.net_transformer import *
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta

def test_net_transformer():
    model = Sequential()
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    #ßprint(model.summary())
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    random_input = np.random.rand(1,28,28,1)#one picture, 28,28, 1 chanel
    output1 = model.predict_on_batch(random_input)
    #print(model.summary())
    models = net_transfromer(model)
    #print(models[5].summary())
    for new_model in models:
        output2 = new_model.predict_on_batch(random_input)
        assert np.sum(output1.flatten() - output2.flatten()) < 1e-4