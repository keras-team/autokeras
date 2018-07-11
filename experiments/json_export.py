import numpy as np
from autokeras import ImageClassifier
from tests.common import clean_dir

if __name__ == '__main__':
    x_train = np.random.rand(2, 28, 28, 1)
    y_train = np.random.rand(2)
    x_test = np.random.rand(1, 28, 28, 1)
    y_test = np.random.rand(1)
    trainer_args = {'max_iter_num': 0,
                    'batch_size': 128,
                    'augment': False}
    clf = ImageClassifier(path='tests/resources/temp', verbose=True, searcher_args={'trainer_args': trainer_args})

    clf.fit(x_train, y_train, time_limit=30)
    clf.load_searcher().export_json('./test.json')
    clean_dir('tests/resources/temp')
