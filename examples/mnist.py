from tensorflow.keras.datasets import mnist

import autokeras as ak


def task_api():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    clf = ak.ImageClassifier(seed=5, max_trials=1)
    clf.fit(x_train, y_train, validation_split=0.2, epochs=1)
    ak.utils.pickle_to_file(
        clf.tuner.oracle.get_best_trials(1)[0].hyperparameters.values,
        '/tmp/mnist_best_hps')
    return clf.evaluate(x_test, y_test)


def io_api():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    clf = ak.AutoModel(ak.ImageInput(),
                       ak.ClassificationHead(),
                       seed=5,
                       max_trials=3)
    clf.fit(x_train, y_train, validation_split=0.2)
    return clf.evaluate(x_test, y_test)


def functional_api():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    input_node = ak.ImageInput()
    output_node = input_node
    output_node = ak.Normalization()(output_node)
    output_node = ak.ConvBlock()(output_node)
    output_node = ak.SpatialReduction()(output_node)
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.ClassificationHead()(output_node)
    clf = ak.AutoModel(input_node, output_node, seed=5, max_trials=3)
    clf.fit(x_train, y_train, validation_split=0.2)
    return clf.evaluate(x_test, y_test)


if __name__ == '__main__':
    task_api()
