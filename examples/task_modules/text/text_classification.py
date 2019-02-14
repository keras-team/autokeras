from autokeras.text.text_supervised import TextClassifier
from autokeras.utils import read_tsv_file


if __name__ == '__main__':
    file_path1 = "examples/task_modules/text/train_data.tsv"
    file_path2 = "examples/task_modules/text/test_data.tsv"
    x_train, y_train = read_tsv_file(input_file=file_path1)
    x_test, y_test = read_tsv_file(input_file=file_path2)

    clf = TextClassifier(verbose=True)
    clf.fit(x=x_train, y=y_train, time_limit=12 * 60 * 60)
    print("Classification accuracy is : ", 100 * clf.evaluate(x_test, y_test), "%")
