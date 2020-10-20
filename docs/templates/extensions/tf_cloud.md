# TensorFlow Cloud

[TensorFlow Cloud](https://github.com/tensorflow/cloud) allows you to run your 
TensorFlow program leveraging the computing power on Google Cloud easily.
Please follow the [instructions](https://github.com/tensorflow/cloud) to setup your account.

AutoKeras has successfully integrated with this service.
Now you can run your program on Google Cloud only by inserting a few more lines of code.
Please see the example below.

```python
import argparse
import os

import autokeras as ak
import tensorflow_cloud as tfc
from tensorflow.keras.datasets import mnist


parser = argparse.ArgumentParser(description="Model save path arguments.")
parser.add_argument("--path", required=True, type=str, help="Keras model save path")
args = parser.parse_args()

tfc.run(
    chief_config=tfc.COMMON_MACHINE_CONFIGS["V100_1X"],
    docker_base_image="haifengjin/autokeras:1.0.3",
)

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)

# Initialize the ImageClassifier.
clf = ak.ImageClassifier(max_trials=2)
# Search for the best model.
clf.fit(x_train, y_train, epochs=10)
# Evaluate on the testing data.
print("Accuracy: {accuracy}".format(accuracy=clf.evaluate(x_test, y_test)[1]))

clf.export_model().save(os.path.join(args.path, "model.h5"))
```
You can find the code above [here](https://github.com/tensorflow/cloud/blob/master/tensorflow_cloud/python/tests/integration/call_run_within_script_with_autokeras.py)
