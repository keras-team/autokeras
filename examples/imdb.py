import numpy as np
import tensorflow as tf

import autokeras as ak


def imdb_raw():
    max_features = 20000
    index_offset = 3  # word index offset

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=max_features,
        index_from=index_offset)
    x_train = x_train
    y_train = y_train.reshape(-1, 1)
    x_test = x_test
    y_test = y_test.reshape(-1, 1)

    word_to_id = tf.keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}
    x_train = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_train))
    x_test = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_test))
    x_train = np.array(x_train, dtype=np.str)
    x_test = np.array(x_test, dtype=np.str)
    return (x_train, y_train), (x_test, y_test)


def task_api():
    (x_train, y_train), (x_test, y_test) = imdb_raw()
    clf = ak.TextClassifier(max_trials=3, seed=5)
    clf.fit(x_train, y_train, validation_split=0.2)
    return clf.evaluate(x_test, y_test)


def functional_api():
    max_features = 20000
    max_words = 400
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=max_features,
        index_from=3)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=max_words)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_words)
    print(x_train.dtype)
    print(x_train[:10])
    input_node = ak.Input()
    output_node = input_node
    output_node = ak.EmbeddingBlock(max_features=max_features)(output_node)
    output_node = ak.ConvBlock()(output_node)
    output_node = ak.SpatialReduction()(output_node)
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.ClassificationHead()(output_node)
    clf = ak.AutoModel(input_node, output_node, seed=5, max_trials=3)
    clf.fit(x_train, y_train, validation_split=0.2)
    return clf.evaluate(x_test, y_test)


if __name__ == '__main__':
    task_api()
    # functional_api()
