You can easily export your model the best model found by AutoKeras as a Keras Model.

The following example uses [ImageClassifier](/image_classifier) as an example.
All the tasks and the [AutoModel](/auto_model/#automodel-class) has this [export_model](/auto_model/#export_model-method) function.

```python
from tensorflow.keras.datasets import mnist
import autokeras as ak

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize the image classifier.
clf = ak.ImageClassifier(max_trials=10) # It tries 10 different models.
# Feed the image classifier with training data.
clf.fit(x_train, y_train)
# Export as a Keras Model.
model = clf.export_model()

print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>
```
