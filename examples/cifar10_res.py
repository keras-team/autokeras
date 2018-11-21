import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.datasets import cifar10

from autokeras.nn.generator import ResNetGenerator
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.nn.model_trainer import ModelTrainer
from autokeras.preprocessor import ImageDataTransformer, OneHotEncoder
from autokeras.utils import temp_path_generator, rand_temp_folder_generator

from examples.mixup import main as mixup_main


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    data_transformer = ImageDataTransformer(x_train)
    y_encoder = OneHotEncoder()
    y_encoder.fit(y_train)
    y_train = y_encoder.transform(y_train)
    y_test = y_encoder.transform(y_test)

    train_data = data_transformer.transform_train(x_train, y_train)
    test_data = data_transformer.transform_test(x_test, y_test)

    resnet = ResNetGenerator(10, (32, 32, 3)).generate(5, 64)
    model = resnet.produce_model()
    mixup_main(model)
    # loss, metric_value = ModelTrainer(model=model,
    #                                   path=rand_temp_folder_generator(),
    #                                   train_data=train_data,
    #                                   test_data=test_data,
    #                                   metric=Accuracy,
    #                                   loss_function=classification_loss,
    #                                   verbose=True).train_model(max_iter_num=200, max_no_improvement_num=200)
    # print(loss, metric_value)


if __name__ == '__main__':
    main()
