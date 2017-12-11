import numpy as np


class OneHotEncoder:
    def __init__(self):
        self.data = None
        self.n_classes = 0
        self.labels = None
        self.label_to_vec = {}
        self.int_to_label = {}

    def fit(self, data):
        data = np.array(data).flatten()
        self.labels = set(data)
        self.n_classes = len(self.labels)
        for index, label in enumerate(self.labels):
            vec = np.array([0] * self.n_classes)
            vec[index] = 1
            self.label_to_vec[label] = vec
            self.int_to_label[index] = label

    def transform(self, data):
        data = np.array(data)
        return np.array(list(map(lambda x: self.label_to_vec[x], data)))

    def inverse_transform(self, data):
        return np.array(list(map(lambda x: self.int_to_label[x], np.argmax(np.array(data), axis=1))))
