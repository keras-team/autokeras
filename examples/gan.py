import numpy as np
import tensorflow
from keras.datasets import cifar10

from autokeras.gan import DCGAN

if __name__ == '__main__':
    print(tensorflow.__version__)
    (x_train, _), (_, _) = cifar10.load_data()
    dcgan = DCGAN(ngf=64, ndf=64, nc=3, verbose=True, augment=False, gen_training_result=('image', 32))
    dcgan.fit(x_train)
    output_size = 32

    noise_sample = np.random.randn(output_size, 100, 1, 1).astype('float32')
    dcgan.generate(output_size)
