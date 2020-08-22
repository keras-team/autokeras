import timeit

from tensorflow.keras.datasets import cifar10

import autokeras as ak


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    clf = ak.ImageClassifier(max_trials=10,
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
