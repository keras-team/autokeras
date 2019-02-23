import os

from autokeras.constant import Constant
from autokeras.text.text_supervised import TextClassifier
from autokeras.utils import read_tsv_file, temp_path_generator


def test_text_classifier():
    model_file = os.path.join(temp_path_generator(), 'bert_classifier/pytorch_model.bin')
    if os.path.exists(model_file):
        os.remove(model_file)

    file_path1 = "examples/task_modules/text/train_data.tsv"
    file_path2 = "examples/task_modules/text/test_data.tsv"
    x_train, y_train = read_tsv_file(input_file=file_path1)
    x_train, y_train = x_train[:1], y_train[:1]
    x_test, y_test = read_tsv_file(input_file=file_path2)
    x_test, y_test = x_test[:1], y_test[:1]

    Constant.BERT_TRAINER_BATCH_SIZE = 1
    Constant.BERT_TRAINER_EPOCHS = 1

    clf = TextClassifier(verbose=True)
    clf.fit(x=x_train, y=y_train, time_limit=12 * 60 * 60)
    y_pred = clf.predict(x_test)
    if len(y_pred) != len(y_test):
        raise AssertionError()
