from keras.datasets import mnist
from autokeras import CnnModule
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.preprocessor import ImageDataTransformer, OneHotEncoder


def transform_y(y_train):
    # Transform y_train.
    y_encoder = OneHotEncoder()
    y_encoder.fit(y_train)
    y_train = y_encoder.transform(y_train)
    return y_train, y_encoder


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    y_train, y_encoder = transform_y(y_train)
    y_test, _ = transform_y(y_test)
    cnnModule = CnnModule(loss=classification_loss, metric=Accuracy, searcher_args={}, verbose=True)
    # specify the fit args
    data_transformer = ImageDataTransformer(x_train, augment=True)
    train_data = data_transformer.transform_train(x_train, y_train)
    test_data = data_transformer.transform_test(x_test, y_test)
    fit_args = {
        "n_output_node": y_encoder.n_classes,
        "input_shape": x_train.shape,
        "train_data": train_data,
        "test_data": test_data
    }
    cnnModule.fit(n_output_node=fit_args.get("n_output_node"),
                  input_shape=fit_args.get("input_shape"),
                  train_data=fit_args.get("train_data"),
                  test_data=fit_args.get("test_data"),
                  time_limit=24 * 60 * 60)
