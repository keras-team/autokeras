import os

from keras.datasets import mnist

from autokeras import ImageClassifier
from autokeras.utils import pickle_from_file

# Customer temp dir by your own
TEMP_DIR = '/tmp/autokeras_U8KEOQ'
model_file_name = os.path.join(TEMP_DIR, 'test_autokeras_model.pkl')

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    clf = ImageClassifier(verbose=True, augment=False, path=TEMP_DIR, resume=True)
    clf.fit(x_train, y_train, time_limit=30 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test)
    clf.export_autokeras_model(model_file_name)
    model = pickle_from_file(model_file_name)
    results = model.evaluate(x_test, y_test)
    print(results)
