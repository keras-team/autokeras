This tutorial covers how to do image classification with AutoKeras.
The first step is to prepare your data.

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
```

As you can see, the images
