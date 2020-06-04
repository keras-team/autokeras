"""
Regression tasks estimate a numeric variable, such as the price of a house or voter
turnout.

This example is adapted from a
[notebook](https://gist.github.com/mapmeld/98d1e9839f2d1f9c4ee197953661ed07) which
estimates a person's age from their image, trained on the
[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) photographs of famous
people.

First, prepare your image data in a numpy.ndarray or tensorflow.Dataset format. Each
image must have the same shape, meaning each has the same width, height, and color
channels as other images in the set.
"""

"""
### Connect your Google Drive for Data
"""


import os
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from google.colab import drive
from PIL import Image
from scipy.io import loadmat
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak
drive.mount("/content/drive")

"""
### Install AutoKeras and TensorFlow

Download the master branch to your Google Drive for this tutorial. In general, you can
use *pip install autokeras* .
"""

"""shell
!pip install  -v "/content/drive/My Drive/AutoKeras-dev/autokeras-master.zip"
!pip uninstall keras-tuner
!pip install
git+git://github.com/keras-team/keras-tuner.git@d2d69cba21a0b482a85ce2a38893e2322e139c01
"""

"""shell
!pip install tensorflow==2.2.0
"""

"""
###**Import IMDB Celeb images and metadata**
"""

"""shell
!mkdir ./drive/My\ Drive/mlin/celebs
"""

"""shell
! wget -O ./drive/My\ Drive/mlin/celebs/imdb_0.tar
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_0.tar
"""

"""shell
! cd ./drive/My\ Drive/mlin/celebs && tar -xf imdb_0.tar
! rm ./drive/My\ Drive/mlin/celebs/imdb_0.tar
"""

"""
Uncomment and run the below cell if you need to re-run the cells again and above don't
need to install everything from the beginning.
"""

# ! cd ./drive/My\ Drive/mlin/celebs.

"""shell
! ls ./drive/My\ Drive/mlin/celebs/imdb/
"""

"""shell
! wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar
! tar -xf imdb_meta.tar
! rm imdb_meta.tar
"""

"""
###**Converting from MATLAB date to actual Date-of-Birth**
"""


def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    try:
        return (
            datetime.fromordinal(int(datenum))
            + timedelta(days=int(days))
            + timedelta(hours=int(hours))
            + timedelta(minutes=int(minutes))
            + timedelta(seconds=round(seconds))
            - timedelta(days=366)
        )
    except:
        return datenum_to_datetime(700000)


print(datenum_to_datetime(734963))

"""
### **Opening MatLab file to Pandas DataFrame**
"""


x = loadmat("imdb/imdb.mat")


mdata = x["imdb"]  # variable in mat file
mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
ndata = {n: mdata[n][0, 0] for n in mdtype.names}
columns = [n for n, v in ndata.items()]

rows = []
for col in range(0, 10):
    values = list(ndata.items())[col]
    for num, val in enumerate(values[1][0], start=0):
        if col == 0:
            rows.append([])
        if num > 0:
            if columns[col] == "dob":
                rows[num].append(datenum_to_datetime(int(val)))
            elif columns[col] == "photo_taken":
                rows[num].append(datetime(year=int(val), month=6, day=30))
            else:
                rows[num].append(val)

dt = map(lambda row: np.array(row), np.array(rows[1:]))

df = pd.DataFrame(data=dt, index=range(0, len(rows) - 1), columns=columns)
print(df.head())

print(columns)
print(df["full_path"])

"""
### **Calculating age at time photo was taken**
"""

df["age"] = (df["photo_taken"] - df["dob"]).astype("int") / 31558102e9
print(df["age"])

"""
### **Creating dataset**


* We sample 200 of the images which were included in this first download.
* Images are resized to 128x128 to standardize shape and conserve memory
* RGB images are converted to grayscale to standardize shape
* Ages are converted to ints


"""


def df2numpy(train_set):
    images = []
    for img_path in train_set["full_path"]:
        img = (
            Image.open("./drive/My Drive/mlin/celebs/imdb/" + img_path[0])
            .resize((128, 128))
            .convert("L")
        )
        images.append(np.asarray(img, dtype="int32"))

    image_inputs = np.array(images)

    ages = train_set["age"].astype("int").to_numpy()
    return image_inputs, ages


train_set = df[df["full_path"] < "02"].sample(200)
train_imgs, train_ages = df2numpy(train_set)

test_set = df[df["full_path"] < "02"].sample(100)
test_imgs, test_ages = df2numpy(test_set)

"""
### **Training using AutoKeras**
"""


# Initialize the image regressor
reg = ak.ImageRegressor(max_trials=15)  # AutoKeras tries 15 different models.

# Find the best model for the given training data
reg.fit(train_imgs, train_ages)

# Predict with the chosen model:
# predict_y = reg.predict(test_images)  # Uncomment if required

# Evaluate the chosen model with testing data
print(reg.evaluate(test_images, test_ages))

"""
### **Validation Data**

By default, AutoKeras use the last 20% of training data as validation data. As shown in
the example below, you can use validation_split to specify the percentage.
"""

reg.fit(
    train_imgs,
    train_ages,
    # Split the training data and use the last 15% as validation data.
    validation_split=0.15,
    epochs=3,
)

"""
You can also use your own validation set instead of splitting it from the training data
with validation_data.
"""

split = 460000
x_val = train_imgs[split:]
y_val = train_ages[split:]
x_train = train_imgs[:split]
y_train = train_ages[:split]
reg.fit(
    x_train,
    y_train,
    # Use your own validation set.
    validation_data=(x_val, y_val),
    epochs=3,
)

"""
### **Customized Search Space**

For advanced users, you may customize your search space by using AutoModel instead of
ImageRegressor. You can configure the ImageBlock for some high-level configurations,
e.g., block_type for the type of neural network to search, normalize for whether to do
data normalization, augment for whether to do data augmentation. You can also choose not
to specify these arguments, which would leave the different choices to be tuned
automatically. See the following example for detail.
"""


input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    # Only search ResNet architectures.
    block_type="resnet",
    # Normalize the dataset.
    normalize=True,
    # Do not do data augmentation.
    augment=False,
)(input_node)
output_node = ak.RegressionHead()(output_node)
reg = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
reg.fit(x_train, y_train, epochs=3)

"""
The usage of AutoModel is similar to the functional API of Keras. Basically, you are
building a graph, whose edges are blocks and the nodes are intermediate outputs of
blocks. To add an edge from input_node to output_node with output_node =
ak.some_block(input_node).
You can even also use more fine grained blocks to customize the search space even
further. See the following example.
"""


input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation(translation_factor=0.3)(output_node)
output_node = ak.ResNetBlock(version="v2")(output_node)
output_node = ak.RegressionHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
clf.fit(x_train, y_train, epochs=3)

"""
### **Data Format**
"""

"""
The AutoKeras ImageClassifier is quite flexible for the data format.

For the image, it accepts data formats both with and without the channel dimension. The
images in the IMDB-Wiki dataset do not have a channel dimension. Each image is a matrix
with shape (128, 128). AutoKeras also accepts images with a channel dimension at last,
e.g., (32, 32, 3), (28, 28, 1).

For the classification labels, AutoKeras accepts both plain labels, i.e. strings or
integers, and one-hot encoded labels, i.e. vectors of 0s and 1s.

So if you prepare your data in the following way, the ImageClassifier should still work.
"""

# Reshape the images to have the channel dimension.
train_imgs = train_imgs.reshape(train_imgs.shape + (1,))
test_imgs = test_imgs.reshape(test_imgs.shape + (1,))

print(train_imgs.shape)  # (200, 128, 128, 1)
print(test_imgs.shape)  # (100, 128, 128, 1)
print(train_ages[:3])

"""
We also support using tf.data.Dataset format for the training data. In this case, the
images would have to be 3-dimentional. The labels have to be one-hot encoded for
multi-class classification to be wrapped into tensorflow Dataset.
"""


train_set = tf.data.Dataset.from_tensor_slices(((train_imgs,), (train_ages,)))
test_set = tf.data.Dataset.from_tensor_slices(((test_imgs,), (test_ages,)))

reg = ak.ImageRegressor(max_trials=15)
# Feed the tensorflow Dataset to the classifier.
reg.fit(train_set)
# Predict with the best model.
predicted_y = clf.predict(test_set)
# Evaluate the best model with testing data.
print(clf.evaluate(test_set))

"""
## References

[Main Reference
Notebook](https://gist.github.com/mapmeld/98d1e9839f2d1f9c4ee197953661ed07),
[Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/),
[ImageRegressor](/image_regressor),
[ResNetBlock](/block/#resnetblock-class),
[ImageInput](/node/#imageinput-class),
[AutoModel](/auto_model/#automodel-class),
[ImageBlock](/block/#imageblock-class),
[Normalization](/preprocessor/#normalization-class),
[ImageAugmentation](/preprocessor/#image-augmentation-class),
[RegressionHead](/head/#regressionhead-class).

"""
