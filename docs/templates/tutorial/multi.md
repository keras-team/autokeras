In this tutorial we are making use of the 
[AutoModel](/auto_model/#automodel-class)
 API to show how to handle multi-modal data and multi-task.

Multi-model data means each data instance has multiple forms of information. For example, a photo can be saved as a image. Besides the image, it may also have when and where it was taken as its attributes, which can be represented as structured data. 

Multi-task here we refer to we want to predict multiple targets with the same input features. For example, we not only want to classify an image according to its content, but we also want to regress its quality as a float number between 0 and 1.

The following diagram shows an example of multi-modal and multi-task neural network model. It has two inputs the images and the structured data. Each image is associated with a set of attributes in the structured data. From these data, we are trying to predict the classification label and the regression value at the same time.

<div class="mermaid">
graph TD
    id1(ImageInput) --> id3(Some Neural Network Model)
    id2(StructuredDataInput) --> id3
    id3 --> id4(ClassificationHead)
    id3 --> id5(RegressionHead)
</div>

To illustrate our idea, we generate some image and structured data as the multi-modal data to make it easier to understand.

```python
num_instances = 100
# Generate image data.
image_data = np.random.rand(num_instances, 32, 32, 3).astype(np.float32)
# Generate structured data.
structured_data = np.random.rand(num_instances, 20).astype(np.float32)
```

We also generate some multi-task targets for classification and regression.

```python
# Generate regression targets.
regression_target = np.random.rand(num_instances, 1).astype(np.float32)
# Generate classification labels of five classes.
classification_target = np.random.randint(5, size=num_instances)
```


## Validation Data

## Customized Search Space
