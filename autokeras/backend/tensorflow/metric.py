from tensorflow.keras.metrics import categorical_accuracy, mean_squared_error, binary_accuracy


def classification_metric(prediction, target):
    return categorical_accuracy


def regression_metric(prediction, target):
    return mean_squared_error


def binary_classification_metric(prediction, target):
    return binary_accuracy
