import numpy as np

from autokeras.text.text_supervised import TextClassifier
from autokeras.utils import read_tsv_file


def convert_labels_to_one_hot(labels, num_labels):
    one_hot = np.zeros((len(labels), num_labels))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


if __name__ == '__main__':
    file_path1 = "examples/task_modules/text/train_data.tsv"
    file_path2 = "examples/task_modules/text/test_data.tsv"
    x_train, y_train = read_tsv_file(input_file=file_path1)
    x_test, y_test = read_tsv_file(input_file=file_path2)

    y_train = convert_labels_to_one_hot(y_train, num_labels=2)
    y_test = convert_labels_to_one_hot(y_test, num_labels=2)

    clf = TextClassifier(verbose=True)
    clf.fit(x=x_train, y=y_train, time_limit=12 * 60 * 60)
    print("Classification accuracy is : ", 100 * clf.evaluate(x_test, y_test), "%")
