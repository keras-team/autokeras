import numpy as np
import os

from autokeras.constant import Constant
from autokeras.text.text_supervised import TextClassifier, TextRegressor
from autokeras.utils import read_tsv_file, temp_path_generator


def convert_labels_to_one_hot(labels, num_labels):
    one_hot = np.zeros((len(labels), num_labels))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def test_text_classifier():
    model_file = os.path.join(temp_path_generator(), 'bert_classifier/pytorch_text_classifier.bin')
    if os.path.exists(model_file):
        os.remove(model_file)

    file_path1 = "examples/task_modules/text/train_data.tsv"
    file_path2 = "examples/task_modules/text/test_data.tsv"
    x_train, y_train = read_tsv_file(input_file=file_path1)
    x_train, y_train = x_train[:2], y_train[:2]
    x_test, y_test = read_tsv_file(input_file=file_path2)
    x_test, y_test = x_test[:2], y_test[:2]

    y_train = convert_labels_to_one_hot(y_train, 2)
    y_test = convert_labels_to_one_hot(y_test, 2)

    Constant.BERT_TRAINER_BATCH_SIZE = 2
    Constant.BERT_TRAINER_EPOCHS = 1

    clf = TextClassifier(verbose=True)
    clf.fit(x=x_train, y=y_train, time_limit=12 * 60 * 60)
    y_pred = clf.predict(x_test)
    if len(y_pred) != len(y_test):
        raise AssertionError()
    clf.evaluate(x_test, y_test)


def test_text_regressor():
    model_file = os.path.join(temp_path_generator(), 'bert_classifier/pytorch_text_regressor.bin')
    if os.path.exists(model_file):
        os.remove(model_file)

    file_path1 = "examples/task_modules/text/train_data.tsv"
    file_path2 = "examples/task_modules/text/test_data.tsv"
    x_train, y_train = read_tsv_file(input_file=file_path1)
    x_train, y_train = x_train[:2], y_train[:2].astype(float)
    x_test, y_test = read_tsv_file(input_file=file_path2)
    x_test, y_test = x_test[:2], y_test[:2].astype(float)

    Constant.BERT_TRAINER_BATCH_SIZE = 2
    Constant.BERT_TRAINER_EPOCHS = 1

    clf = TextRegressor(verbose=True)
    clf.fit(x=x_train, y=y_train, time_limit=12 * 60 * 60)
    y_pred = clf.predict(x_test)
    if len(y_pred) != len(y_test):
        raise AssertionError()
    clf.evaluate(x_test, y_test)
