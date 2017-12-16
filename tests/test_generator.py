from autokeras.generator import *


def test_random_classifier_generator():
    generator = RandomConvClassifierGenerator(3, (28, 28, 1))
    for i in range(3):
        generator.generate()
