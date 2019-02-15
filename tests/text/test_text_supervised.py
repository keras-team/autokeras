import os
import random

from autokeras.text.text_supervised import TextClassifier
from autokeras.utils import read_tsv_file, temp_path_generator, has_file


def test_fit_predict():
    file_path1 = "examples/task_modules/text/train_data.tsv"
    file_path2 = "examples/task_modules/text/test_data.tsv"
    model_path = temp_path_generator() + 'bert_classifier/pytorch_model.bin'
    if has_file(model_path):
        os.remove(model_path)

    x_train, y_train = read_tsv_file(input_file=file_path1)
    assert len(x_train) != 0 and len(y_train) != 0

    xy_train = list(zip(x_train, y_train))
    random.shuffle(xy_train)
    x_train = [x for x, y in xy_train[:200]]
    y_train = [y for x, y in xy_train[:200]]

    x_test, y_test = read_tsv_file(input_file=file_path2)[:50]
    assert len(x_test) != 0 and len(y_test) != 0

    clf = TextClassifier(verbose=True)
    clf.fit(x=x_train, y=y_train, time_limit=12 * 60 * 60)

    labels = clf.predict(["This is a positive statement", "This is a negative statement"])
    assert (labels == [1, 0]).all()

    accuracy = clf.evaluate(x_test, y_test)
    assert accuracy > 0.75
