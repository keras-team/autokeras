"""shell
pip install autokeras
"""
import numpy as np
from keras.datasets import mnist
from keras.models import load_model

import autokeras as ak

"""
You can easily export your model the best model found by AutoKeras as a Keras
Model.

The following example uses [ImageClassifier](/image_classifier) as an example.
All the tasks and the [AutoModel](/auto_model/#automodel-class) has this
[export_model](/auto_model/#export_model-method) function.

"""


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize the image classifier.
clf = ak.ImageClassifier(
    overwrite=True, max_trials=1
)  # Try only 1 model.(Increase accordingly)
# Feed the image classifier with training data.
clf.fit(x_train, y_train, epochs=1)  # Change no of epochs to improve the model
# Export as a Keras Model.
model = clf.export_model()

print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>

try:
    model.save("model_autokeras", save_format="tf")
except Exception:
    model.save("model_autokeras.h5")


loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)

predicted_y = loaded_model.predict(np.expand_dims(x_test, -1))
print(predicted_y)
