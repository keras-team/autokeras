from autokeras.generator import *
from unittest.mock import *
from autokeras import constant
import numpy as np


def test_random_classifier_generator():
    generator = RandomConvClassifierGenerator(3, (28, 28, 1))
    for i in range(3):
        model = generator.generate()
        model.predict_on_batch(np.random.rand(2, 28, 28, 1))


def test_hill_climbing_classifier_generator():
    x_train = np.random.rand(2, 28, 28, 1)
    y_train = np.random.rand(2, 3)
    x_test = np.random.rand(1, 28, 28, 1)
    y_test = np.random.rand(1, 3)

    constant.MAX_ITER_NUM = 1
    constant.MAX_MODEL_NUM = 1
    generator = HillClimbingClassifierGenerator(3, (28, 28, 1), x_train=x_train, y_train=y_train, x_test=x_test,
                                                y_test=y_test, verbose=False)
    model = None
    generate_model = generator.generate()
    times = 1
    while generate_model is not None and times <= constant.MAX_MODEL_NUM:
        model = generate_model
        generate_model = generator.generate()
        times += 1
    print(model.summary())
