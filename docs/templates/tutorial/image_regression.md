# Image Regression
## Celebrity Ages Example

Regression tasks estimate a numeric variable, such as the price of a house
or voter turnout.

This example is adapted from
[a notebook](https://gist.github.com/mapmeld/98d1e9839f2d1f9c4ee197953661ed07)
which estimates a person's age from their image, trained on the
[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
photographs of famous people.

First, prepare your image data in a ```numpy.ndarray``` or ```tensorflow.Dataset```
format. Each image must have the same shape, meaning
each has the same width, height, and color channels as other images in the set.

```python
import pandas as pd
import numpy as np
from PIL import Image

# converting from other formats (such as pandas) to numpy
age_outputs = train_df.age.to_numpy(dtype="int")

# convert file paths into consistent images
images = []
for img_path in train_df.img_path:
  images.append(
    np.asarray(
      Image.open(img_path)
        .resize((256, 128)) # same dimensions on every image
        .convert('L') # grayscale (original data had mix of RGB and grayscale)
    )
  )
image_inputs = np.array(images)
```

Next, initialize and train the [ImageRegressor](/image_regressor).

```python
import autokeras as ak

# Initialize the image regressor
reg = ak.ImageRegressor(max_trials=15) # AutoKeras tries 15 different models.

# Find the best model for the given training data
reg.fit(image_inputs, age_outputs)

# Predict with the chosen model:
predict_y = reg.predict(predict_x)
```

Measure the accuracy of the regressor on an independent test set:

```python
# Evaluate the chosen model with testing data
print(reg.evaluate(test_images, test_ages))
```
