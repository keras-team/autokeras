from sklearn.metrics import accuracy_score, mean_squared_error


def classification_metric(prediction, target):
    prediction = list(map(lambda x: x.argmax(), prediction))
    target = list(map(lambda x: x.argmax(), target))
    return accuracy_score(target, prediction)


def regression_metric(prediction, target):
    return mean_squared_error(target, prediction)


def binary_classification_metric(prediction, target):
    return accuracy_score(target, prediction)
