import timeit

import numpy as np
from tensorflow.keras.datasets import imdb

import autokeras as ak


def imdb_raw():
    index_offset = 3  # word index offset

    (x_train, y_train), (x_test, y_test) = imdb.load_data(index_from=index_offset)

    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}
    x_train = list(
        map(lambda sentence: " ".join(id_to_word[i] for i in sentence), x_train)
    )
    x_test = list(
        map(lambda sentence: " ".join(id_to_word[i] for i in sentence), x_test)
    )
    x_train = np.array(x_train, dtype=np.str)
    x_test = np.array(x_test, dtype=np.str)
    return (x_train, y_train), (x_test, y_test)


def main():
    (x_train, y_train), (x_test, y_test) = imdb_raw()
    clf = ak.TextClassifier(max_trials=10,
                            directory='tmp_dir',
                            overwrite=True)

    start_time = timeit.default_timer()
    clf.fit(x_train, y_train)
    stop_time = timeit.default_timer()

    accuracy = clf.evaluate(x_test, y_test)[1]
    print('Accuracy: {accuracy}%'.format(accuracy=round(accuracy * 100, 2)))
    print('Total time: {time} seconds.'.format(time=round(stop_time - start_time, 2)))


if __name__ == "__main__":
    main()
