from abc import abstractmethod
from autokeras.backend import Backend
from sklearn.metrics import accuracy_score, mean_squared_error


class Metric:

    @classmethod
    @abstractmethod
    def higher_better(cls):
        pass

    @classmethod
    @abstractmethod
    def compute(cls, prediction, target):
        pass

    @classmethod
    @abstractmethod
    def evaluate(cls, prediction, target):
        pass


class Accuracy(Metric):
    @classmethod
    def higher_better(cls):
        return True

    @classmethod
    def compute(cls, prediction, target):
        return Backend.classification_metric(prediction, target)

    @classmethod
    def evaluate(cls, prediction, target):
        return accuracy_score(target, prediction)


class MSE(Metric):
    @classmethod
    def higher_better(cls):
        return False

    @classmethod
    def compute(cls, prediction, target):
        return Backend.regression_metric(prediction, target)

    @classmethod
    def evaluate(cls, prediction, target):
        return mean_squared_error(target, prediction)
