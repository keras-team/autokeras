import torchvision
from keras.datasets import cifar10, mnist
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

from autokeras.generator import DefaultClassifierGenerator
from autokeras.net_transformer import default_transform
from autokeras.preprocessor import OneHotEncoder, DataTransformer
from autokeras.utils import ModelTrainer


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    print('Start Encoding')
    encoder = OneHotEncoder()
    encoder.fit(y_train)

    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    data_transformer = DataTransformer(x_train, augment=False)

    train_data = data_transformer.transform_train(x_train, y_train)
    test_data = data_transformer.transform_test(x_test, y_test)

    print('Start Generating')
    graphs = [DefaultClassifierGenerator(10, x_train.shape[1:]).generate()]
    keras_model = graphs[0].produce_model()


    print('Start Training')
    loss, acc = ModelTrainer(keras_model,
                             train_data,
                             test_data,
                             True).train_model(max_no_improvement_num=100, batch_size=128)
    print(loss, acc)
