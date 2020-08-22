import timeit

import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import autokeras as ak


def main():
    house_dataset = fetch_california_housing()
    data = house_dataset.data
    target = house_dataset.target
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    clf = ak.StructuredDataRegressor(max_trials=10, directory='tmp_dir', overwrite=True)

    start_time = timeit.default_timer()
    clf.fit(x_train, y_train)
    stop_time = timeit.default_timer()

    mse = clf.evaluate(x_test, y_test)[1]
    print('MSE: {mse}'.format(mse=round(mse, 2)))
    print('Total time: {time} seconds.'.format(time=round(stop_time - start_time, 2)))


if __name__ == "__main__":
    main()

