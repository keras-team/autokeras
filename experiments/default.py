from keras.datasets import cifar10

from autokeras.generator import DefaultClassifierGenerator
from autokeras.net_transformer import default_transform
from autokeras.preprocessor import OneHotEncoder
from autokeras.utils import ModelTrainer


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print('Start Encoding')
    encoder = OneHotEncoder()
    encoder.fit(y_train)

    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    print('Start Generating')
    graphs = default_transform(DefaultClassifierGenerator(10, x_train.shape[1:]).generate())
    keras_model = graphs[0].produce_model()

    print('Start Training')
    ModelTrainer(keras_model,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 True).train_model(max_no_improvement_num=100, batch_size=128)
    print(keras_model.evaluate(x_test, y_test, True))
