"""shell
pip install autokeras
pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc2
"""

"""
If the data is too large to put in memory all at once, we can load it batch by batch into memory from disk with tf.data.Dataset.
This [function](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory) can help you build such a tf.data.Dataset for image data.

First, we download the data and extract the files.
"""

import tensorflow as tf
import os

# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# local_file_path = tf.keras.utils.get_file(origin=dataset_url, 
                                          # fname='image_data', 
                                          # extract=True)
# # The file is extracted in the same directory as the downloaded file.
# local_dir_path = os.path.dirname(local_file_path)
# # After check mannually, we know the extracted data is in 'flower_photos'.
# data_dir = os.path.join(local_dir_path, 'flower_photos')
# print(data_dir)

"""
The directory should look like this. Each folder contains the images in the same class.

```
flowers_photos/
  daisy/
  dandelion/
  roses/
  sunflowers/
  tulips/
```

We can split the data into training and testing as we load them.
"""

batch_size = 32
img_height = 180
img_width = 180

# train_data = tf.keras.preprocessing.image_dataset_from_directory(
    # data_dir,
    # # Use 20% data as testing data.
    # validation_split=0.2,
    # subset="training",
    # # Set seed to ensure the same split when loading testing data.
    # seed=123,
    # image_size=(img_height, img_width),
    # batch_size=batch_size)

# test_data = tf.keras.preprocessing.image_dataset_from_directory(
    # data_dir,
    # validation_split=0.2,
    # subset="validation",
    # seed=123,
    # image_size=(img_height, img_width),
    # batch_size=batch_size)

"""
Then we just do one quick demo of AutoKeras to make sure the dataset works.
"""

import autokeras as ak

# clf = ak.ImageClassifier(overwrite=True, max_trials=1)
# clf.fit(train_data, epochs=1)
# print(clf.evaluate(test_data))

"""
You can also load text datasets in the same way.
"""

dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

local_file_path = tf.keras.utils.get_file(
    fname="text_data", 
    origin=dataset_url, 
    extract=True,
)
# The file is extracted in the same directory as the downloaded file.
local_dir_path = os.path.dirname(local_file_path)
# After check mannually, we know the extracted data is in 'aclImdb'.
data_dir = os.path.join(local_dir_path, 'aclImdb')
# Remove the unused data folder.
import shutil
shutil.rmtree(os.path.join(data_dir, 'train/unsup'))


"""
For this dataset, the data is already split into train and test.
We just load them separately.
"""
print(data_dir)
train_data = tf.keras.preprocessing.text_dataset_from_directory(
    os.path.join(data_dir, 'train'),
    class_names=['pos', 'neg'],
    validation_split=0.2,
    subset="training",
    # shuffle=False,
    seed=123,
    batch_size=batch_size)

val_data = tf.keras.preprocessing.text_dataset_from_directory(
    os.path.join(data_dir, 'train'),
    class_names=['pos', 'neg'],
    validation_split=0.2,
    subset="validation",
    # shuffle=False,
    seed=123,
    batch_size=batch_size)

test_data = tf.keras.preprocessing.text_dataset_from_directory(
    os.path.join(data_dir, 'test'),
    class_names=['pos', 'neg'],
    shuffle=False,
    batch_size=batch_size)

for x, y in train_data:
    print(x.numpy()[0])
    print(y.numpy()[0])
    # record_x = x.numpy()
    # record_y = y.numpy()
    break

for x, y in train_data:
    print(x.numpy()[0])
    print(y.numpy()[0])
    break

# train_data = tf.keras.preprocessing.text_dataset_from_directory(
    # os.path.join(data_dir, 'train'),
    # class_names=['pos', 'neg'],
    # shuffle=True,
    # seed=123,
    # batch_size=batch_size)

# for x, y in train_data:
    # for i, a in enumerate(x.numpy()):
        # for j, b in enumerate(record_x):
            # if a == b:
                # print('*')
                # assert record_y[j] == y.numpy()[i]

# import numpy as np
# x_train = []
# y_train = []
# for x, y in train_data:
    # for a in x.numpy():
        # x_train.append(a)
    # for a in y.numpy():
        # y_train.append(a)

# x_train = np.array(x_train)
# y_train = np.array(y_train)

# train_data = train_data.shuffle(1000, seed=123, reshuffle_each_iteration=False)


clf = ak.TextClassifier(overwrite=True, max_trials=2)
# clf.fit(train_data, validation_data=test_data)
# clf.fit(train_data, validation_data=train_data)
clf.fit(train_data, validation_data=val_data)
# clf.fit(x_train, y_train)
# clf.fit(train_data)
print(clf.evaluate(test_data))
