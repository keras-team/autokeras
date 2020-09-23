import timeit
import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from sklearn.datasets import load_files

import autokeras as ak


def imdb_raw():
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz", 
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
        extract=True,
    )
    
    # set path to dataset
    IMDB_DATADIR = os.path.join(os.path.dirname(dataset), 'aclImdb')
    
    classes = ['pos', 'neg']
    train_data = load_files(os.path.join(IMDB_DATADIR, 'train'), shuffle=True, categories=classes)
    test_data = load_files(os.path.join(IMDB_DATADIR,  'test'), shuffle=False, categories=classes)

    encoding = 'utf-8'
    x_train = np.array([x.decode(encoding) for x in train_data.data])
    y_train = np.array(train_data.target)
    x_test = np.array([x.decode(encoding) for x in test_data.data])
    y_test = np.array(test_data.target)
    return (x_train, y_train), (x_test, y_test)


def main():
    (x_train, y_train), (x_test, y_test) = imdb_raw()
    clf = ak.TextClassifier(max_trials=1,
                            directory='tmp_dir',
                            overwrite=True)

    start_time = timeit.default_timer()
    clf.fit(x_train, y_train, batch_size=6, epochs=1, shuffle=True)
    stop_time = timeit.default_timer()

    accuracy = clf.evaluate(x_test, y_test)[1]
    print('Accuracy: {accuracy}%'.format(accuracy=round(accuracy * 100, 2)))
    print('Total time: {time} seconds.'.format(time=round(stop_time - start_time, 2)))


if __name__ == "__main__":
    main()
