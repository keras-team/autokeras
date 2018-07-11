from keras.datasets import mnist
import os

from pandas import DataFrame
from PIL import Image

from autokeras.utils import ensure_dir

ensure_dir('mnist/train')
ensure_dir('mnist/test')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(x_train.shape + (1,))
# x_test = x_test.reshape(x_test.shape + (1,))

# file_names = []
# for i in range(len(x_train)):
#     file_name = ("%05d" % (i, )) + '.jpg'
#     Image.fromarray(x_train[i]).save(os.path.join('mnist', 'train', file_name))
#     file_names.append(file_name)
#
# csv_data = {'File Name': file_names, 'Label': y_train}
# DataFrame(csv_data).to_csv('mnist/train/label.csv', index=False)

file_names = []
for i in range(len(x_test)):
    file_name = ("%05d" % (i, )) + '.jpg'
    Image.fromarray(x_test[i]).save(os.path.join('mnist', 'test', file_name))
    file_names.append(file_name)

csv_data = {'File Name': file_names, 'Label': y_test}
DataFrame(csv_data).to_csv('mnist/test/label.csv', index=False)
