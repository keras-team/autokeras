import pandas as pd

from autokeras import TextClassifier


def read_csv(file_path):
    """csv file read example method
    It helps you to read the csv file into python array

    Attributes:
        file_path: csv file path
    """

    print("reading data...")
    data_train = pd.read_csv(file_path, sep='\t')

    x_train = []
    y_train = []
    for idx in range(data_train.review.shape[0]):
        # Modify this according to each different dataset
        x_train.append(data_train.review[idx])
        y_train.append(data_train.sentiment[idx])
    return x_train, y_train


if __name__ == '__main__':
    file_path = "labeledTrainData.tsv"
    x_train, y_train = read_csv(file_path=file_path)
    clf = TextClassifier(verbose=True)
    clf.fit(x=x_train, y=y_train, time_limit=12 * 60 * 60)
    clf.final_fit(x_train=x_train, y_train=y_train, retrain=True)
