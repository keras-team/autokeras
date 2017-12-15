from autokeras.comparator import *

def test_comparator_one():
    model_a = Sequential()

    model_a.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))
    model_a.add(Conv2D(32, (3, 3), activation='relu'))
    model_a.add(MaxPooling2D(pool_size=(2, 2)))
    model_a.add(Dropout(0.25))

    model_a.add(Flatten())
    model_a.add(Dense(128, activation='relu'))
    model_a.add(Dropout(0.5))
    model_a.add(Dense(10, activation='softmax'))

    model_b = Sequential()

    model_b.add(Conv2D(32, (3, 3), activation='relu', input_shape=( 28, 28,1)))
    model_b.add(Conv2D(32, (3, 3), activation='relu'))
    model_b.add(MaxPooling2D(pool_size=(2, 2)))
    model_b.add(Dropout(0.25))

    model_b.add(Flatten())
    model_b.add(Dense(128, activation='relu'))
    model_b.add(Dropout(0.5))
    model_b.add(Dense(10, activation='softmax'))

    assert compare_network(model_a,model_b) == True

def test_comparator_two():
    model_a = Sequential()

    model_a.add(Conv2D(32, (3, 3), activation='softmax', input_shape=(28, 28,1)))
    model_a.add(Conv2D(32, (3, 3), activation='relu'))
    model_a.add(MaxPooling2D(pool_size=(2, 2)))
    model_a.add(Dropout(0.25))

    model_a.add(Flatten())
    model_a.add(Dense(128, activation='relu'))
    model_a.add(Dropout(0.5))
    model_a.add(Dense(10, activation='softmax'))

    model_b = Sequential()

    model_b.add(Conv2D(32, (3, 3), activation='relu', input_shape=( 28, 28,1)))
    model_b.add(Conv2D(32, (3, 3), activation='relu'))
    model_b.add(MaxPooling2D(pool_size=(2, 2)))
    model_b.add(Dropout(0.25))

    model_b.add(Flatten())
    model_b.add(Dense(128, activation='relu'))
    model_b.add(Dropout(0.5))
    model_b.add(Dense(10, activation='softmax'))
    assert compare_network(model_a,model_b) == False


