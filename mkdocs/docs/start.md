# Getting Started

## Installation
The installation of Auto-Keras is the same as other python packages.
Just use pip install.
You can run the following command in your terminal.

    pip install autokeras

## Example

Here we show an example of image classification on the MNIST dataset, is a famous image dataset for hand-written digits classification.

    from keras.datasets import mnist
    from autokeras.classifier import ImageClassifier

    if __name__ == '__main__':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))

        clf = ImageClassifier(verbose=True)
        clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
        clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
        y = clf.evaluate(x_test, y_test)
        print(y)
