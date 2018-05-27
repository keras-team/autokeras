import numpy as np

from autokeras.generator import *


def test_random_classifier_generator():
    generator = RandomConvClassifierGenerator(3, (28, 28, 1))
    for i in range(3):
        model = generator.generate()
        model.predict_on_batch(np.random.rand(2, 28, 28, 1))


def test_default_generator():
    generator = DefaultClassifierGenerator(3, (28, 28, 1))
    generator.generate()
