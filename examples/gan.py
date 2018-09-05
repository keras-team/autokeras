from keras.datasets import cifar10
from autokeras.gan import DCGAN
import tensorflow

if __name__ == '__main__':
    print(tensorflow.__version__)
    (x_train, _), (_, _) = cifar10.load_data()

    clf = DCGAN(verbose=True, augment=False, gen_training_result=('image', 32))
    clf.train(x_train)
    clf.evaluate(32)
