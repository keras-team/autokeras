import numpy as np
import tensorflow as tf


def imdb_raw(num_instances=100):
    index_offset = 3  # word index offset

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=1000,
        index_from=index_offset)
    x_train = x_train[:num_instances]
    y_train = y_train[:num_instances].reshape(-1, 1)
    x_test = x_test[:num_instances]
    y_test = y_test[:num_instances].reshape(-1, 1)

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
    pass


if __name__ == '__main__':
    task_api()
