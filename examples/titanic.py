"""
Search for a good model for the [Titanic](https://www.kaggle.com/c/titanic)
dataset.
"""

import timeit

import keras

import autokeras as ak

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"


def main():
    # Initialize the classifier.
    train_file_path = keras.utils.get_file("train.csv", TRAIN_DATA_URL)
    test_file_path = keras.utils.get_file("eval.csv", TEST_DATA_URL)
    clf = ak.StructuredDataClassifier(
        max_trials=10, directory="tmp_dir", overwrite=True
    )

    start_time = timeit.default_timer()
    # x is the path to the csv file. y is the column name of the column to
    # predict.
    clf.fit(train_file_path, "survived")
    stop_time = timeit.default_timer()

    # Evaluate the accuracy of the found model.
    accuracy = clf.evaluate(test_file_path, "survived")[1]
    print("Accuracy: {accuracy}%".format(accuracy=round(accuracy * 100, 2)))
    print(
        "Total time: {time} seconds.".format(
            time=round(stop_time - start_time, 2)
        )
    )


if __name__ == "__main__":
    main()
