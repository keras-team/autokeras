In this tutorial we are making use of the 
[AutoModel](/auto_model/#automodel-class)
 API to show how to handle multi-modal data and multi-task.

## What is multi-modal?

Multi-model data means each data instance has multiple forms of information. For example, a photo can be saved as a image. Besides the image, it may also have when and where it was taken as its attributes, which can be represented as structured data. 

## What is multi-task?

Multi-task here we refer to we want to predict multiple targets with the same input features. For example, we not only want to classify an image according to its content, but we also want to regress its quality as a float number between 0 and 1.

The following diagram shows an example of multi-modal and multi-task neural network model.

<div class="mermaid">
graph TD
    id1(ImageInput) --> id3(Some Neural Network Model)
    id2(StructuredDataInput) --> id3
    id3 --> id4(ClassificationHead)
    id3 --> id5(RegressionHead)
</div>

It has two inputs the images and the structured data. Each image is associated with a set of attributes in the structured data. From these data, we are trying to predict the classification label and the regression value at the same time.

## Data Preparation

To illustrate our idea, we generate some random image and structured data as the multi-modal data.

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

## Build and Train the Model
Then we initialize the multi-modal and multi-task model with 
[AutoModel](/auto_model/#automodel-class).

```python
import autokeras as ak
# Initialize the multi with multiple inputs and outputs.
model = ak.AutoModel(
    inputs=[ak.ImageInput(), ak.StructuredDataInput()],
    outputs=[
        ak.RegressionHead(metrics=['mae']),
        ak.ClassificationHead(loss='categorical_crossentropy', metrics=['accuracy'])
    ],
    max_trials=10)
# Fit the model with prepared data.
model.fit(
    [image_data, structured_data],
    [regression_target, classification_target],
    epochs=10)
```

## Validation Data
By default, AutoKeras use the last 20% of training data as validation data.
As shown in the example below, you can use `validation_split` to specify the percentage.

```python
model.fit(
    [image_data, structured_data],
    [regression_target, classification_target],
    # Split the training data and use the last 15% as validation data.
    validation_split=0.15)
```

You can also use your own validation set
instead of splitting it from the training data with `validation_data`.

```python
split = 20

image_val = image_data[split:]
structured_val = structured_data[split:]
regression_val = regression_target[split:]
classification_val = classification_target[split:]

image_data = image_data[:split]
structured_data = structured_data[:split]
regression_target = regression_target[:split]
classification_target = classification_target[:split]

model.fit(x_train,
        y_train,
        # Use your own validation set.
        validation_data=(x_val, y_val))
```

You can customize your search space follow [this tutorial](/tutorial/customized).

## Reference
[AutoModel](/auto_model/#automodel-class),
[ImageInput](/node/#imageinput-class),
[StructuredDataInput](/node/#structureddatainput-class),
[RegressionHead](/head/#regressionhead-class).
[ClassificationHead](/head/#classificationhead-class).
