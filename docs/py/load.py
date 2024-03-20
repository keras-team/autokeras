"""shell
pip install autokeras
"""

import os
import shutil

import keras
import numpy as np
import tensorflow as tf

import autokeras as ak

"""
## Load Images from Disk
If the data is too large to put in memory all at once, we can load it batch by
batch into memory from disk with tf.data.Dataset.  This
[function](
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/
image_dataset_from_directory)
can help you build such a tf.data.Dataset for image data.

First, we download the data and extract the files.
"""

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"  # noqa: E501
local_file_path = keras.utils.get_file(
    origin=dataset_url, fname="image_data", extract=True
)
# The file is extracted in the same directory as the downloaded file.
local_dir_path = os.path.dirname(local_file_path)
# After check mannually, we know the extracted data is in 'flower_photos'.
data_dir = os.path.join(local_dir_path, "flower_photos")
print(data_dir)

"""
The directory should look like this. Each folder contains the images in the
same class.

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

batch_size = 2
img_height = 180
img_width = 180

train_data = ak.image_dataset_from_directory(
    data_dir,
    # Use 20% data as testing data.
    validation_split=0.2,
    subset="training",
    # Set seed to ensure the same split when loading testing data.
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

test_data = ak.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

"""
Then we just do one quick demo of AutoKeras to make sure the dataset works.
"""

clf = ak.ImageClassifier(overwrite=True, max_trials=1)
clf.fit(train_data.take(100), epochs=1)
print(clf.evaluate(test_data.take(2)))

"""
## Load Texts from Disk
You can also load text datasets in the same way.
"""

dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

local_file_path = keras.utils.get_file(
    fname="text_data",
    origin=dataset_url,
    extract=True,
)
# The file is extracted in the same directory as the downloaded file.
local_dir_path = os.path.dirname(local_file_path)
# After check mannually, we know the extracted data is in 'aclImdb'.
data_dir = os.path.join(local_dir_path, "aclImdb")
# Remove the unused data folder.

shutil.rmtree(os.path.join(data_dir, "train/unsup"))


"""
For this dataset, the data is already split into train and test.
We just load them separately.
"""

print(data_dir)
train_data = ak.text_dataset_from_directory(
    os.path.join(data_dir, "train"), batch_size=batch_size
)

test_data = ak.text_dataset_from_directory(
    os.path.join(data_dir, "test"), shuffle=False, batch_size=batch_size
)

clf = ak.TextClassifier(overwrite=True, max_trials=1)
clf.fit(train_data.take(2), epochs=1)
print(clf.evaluate(test_data.take(2)))


"""
## Load Data with Python Generators
If you want to use generators, you can refer to the following code.
"""


N_BATCHES = 2
BATCH_SIZE = 10


def get_data_generator(n_batches, batch_size):
    """Get a generator returning n_batches random data."""

    def data_generator():
        for _ in range(n_batches * batch_size):
            x = np.random.randn(32, 32, 3)
            y = x.sum() / 32 * 32 * 3 > 0.5
            yield x, y

    return data_generator


dataset = tf.data.Dataset.from_generator(
    get_data_generator(N_BATCHES, BATCH_SIZE),
    output_types=(tf.float32, tf.float32),
    output_shapes=((32, 32, 3), tuple()),
).batch(BATCH_SIZE)

clf = ak.ImageClassifier(overwrite=True, max_trials=1, seed=5)
clf.fit(x=dataset, validation_data=dataset, batch_size=BATCH_SIZE)
print(clf.evaluate(dataset))

"""
## Reference
[image_dataset_from_directory](utils/#image_dataset_from_directory-function)
[text_dataset_from_directory](utils/#text_dataset_from_directory-function)
"""
