from autokeras.utils import *
from autokeras.generator import *
import numpy as np
def test_model_rainer():
    model = RandomConvClassifierGenerator(3, (28, 28, 1)).generate()
    ModelTrainer(model,  np.random.rand(2,28,28,1),  np.random.rand(2,3),  np.random.rand(1,28,28,1),
                 np.random.rand(1,3), True).train_model()