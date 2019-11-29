from tensorflow.keras.datasets import cifar10

import autokeras as ak


def task_api():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    clf = ak.ImageClassifier(seed=5, max_trials=10)
    clf.fit(x_train, y_train, validation_split=0.2)
    return clf.evaluate(x_test, y_test)


def io_api():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    clf = ak.AutoModel(ak.ImageInput(),
                       ak.ClassificationHead(),
                       seed=5,
                       max_trials=3)
    clf.fit(x_train, y_train, validation_split=0.2)
    return clf.evaluate(x_test, y_test)


def functional_api():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_node = ak.ImageInput()
    output_node = input_node
    output_node = ak.Normalization()(output_node)
    output_node = ak.ImageAugmentation()(output_node)
    output_node = ak.ResNetBlock(version='next')(output_node)
    output_node = ak.SpatialReduction()(output_node)
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.ClassificationHead()(output_node)
    clf = ak.GraphAutoModel(input_node, output_node, seed=5, max_trials=3)
    clf.fit(x_train, y_train, validation_split=0.2)
    return clf.evaluate(x_test, y_test)


if __name__ == '__main__':
    functional_api()
