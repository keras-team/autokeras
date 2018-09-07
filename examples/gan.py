from keras.datasets import cifar10

from autokeras.gan import DCGAN

if __name__ == '__main__':
    output_size = 32
    (x_train, _), (_, _) = cifar10.load_data()
    # x_train = x_train.reshape(x_train.shape + (1,))
    dcgan = DCGAN(ngf=64, ndf=64, nc=3, verbose=True, augment=False, gen_training_result=('image', output_size))
    dcgan.fit(x_train)
    dcgan.generate()
