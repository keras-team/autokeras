In this tutorial we are making use of the 
[AutoModel](/auto_model/#automodel-class)
 API to show how to handle multi-modal data with multi-task.

## Multi-Modal
Multi-model data means each data instance has multiple forms of information. For example, a photo can be saved as a image. Besides the image, it may also have when and where it was taken as its attributes, which can be represented as structured data. The following examples shows how can we deal with such data.

To illustrate our idea, we need to prepare the data first. We use generated data to make the process easier to understand. We use image data and structured data as the multi-modal data.
```python
num_instances = 100
image_shape = (32, 32, 3)
data = np.random.rand(*((num_instances,) + shape))
if data.dtype == np.float64:
    data = data.astype(np.float32)
```
## Multi-Task
```python
```

