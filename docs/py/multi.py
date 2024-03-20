"""shell
pip install autokeras
"""

import numpy as np

import autokeras as ak

"""
In this tutorial we are making use of the
[AutoModel](/auto_model/#automodel-class)
 API to show how to handle multi-modal data and multi-task.

## What is multi-modal?

Multi-modal data means each data instance has multiple forms of information.
For example, a photo can be saved as a image. Besides the image, it may also
have when and where it was taken as its attributes, which can be represented as
numerical data.

## What is multi-task?

Multi-task here we refer to we want to predict multiple targets with the same
input features. For example, we not only want to classify an image according to
its content, but we also want to regress its quality as a float number between
0 and 1.

The following diagram shows an example of multi-modal and multi-task neural
network model.

<div class="mermaid">
graph TD
    id1(ImageInput) --> id3(Some Neural Network Model)
    id2(Input) --> id3
    id3 --> id4(ClassificationHead)
    id3 --> id5(RegressionHead)
</div>

It has two inputs the images and the numerical input data. Each image is
associated with a set of attributes in the numerical input data. From these
data, we are trying to predict the classification label and the regression value
at the same time.

## Data Preparation

To illustrate our idea, we generate some random image and numerical data as
the multi-modal data.
"""


num_instances = 10
# Generate image data.
image_data = np.random.rand(num_instances, 32, 32, 3).astype(np.float32)
# Generate numerical data.
numerical_data = np.random.rand(num_instances, 20).astype(np.float32)

"""
We also generate some multi-task targets for classification and regression.
"""

# Generate regression targets.
regression_target = np.random.rand(num_instances, 1).astype(np.float32)
# Generate classification labels of five classes.
classification_target = np.random.randint(5, size=num_instances)

"""
## Build and Train the Model
Then we initialize the multi-modal and multi-task model with
[AutoModel](/auto_model/#automodel-class).
Since this is just a demo, we use small amount of `max_trials` and `epochs`.
"""


# Initialize the multi with multiple inputs and outputs.
model = ak.AutoModel(
    inputs=[ak.ImageInput(), ak.Input()],
    outputs=[
        ak.RegressionHead(metrics=["mae"]),
        ak.ClassificationHead(
            loss="categorical_crossentropy", metrics=["accuracy"]
        ),
    ],
    overwrite=True,
    max_trials=2,
)
# Fit the model with prepared data.
model.fit(
    [image_data, numerical_data],
    [regression_target, classification_target],
    epochs=1,
    batch_size=3,
)

"""
## Validation Data
By default, AutoKeras use the last 20% of training data as validation data.
As shown in the example below, you can use `validation_split` to specify the
percentage.
"""

model.fit(
    [image_data, numerical_data],
    [regression_target, classification_target],
    # Split the training data and use the last 15% as validation data.
    validation_split=0.15,
    epochs=1,
    batch_size=3,
)

"""
You can also use your own validation set
instead of splitting it from the training data with `validation_data`.
"""

split = 5

image_val = image_data[split:]
numerical_val = numerical_data[split:]
regression_val = regression_target[split:]
classification_val = classification_target[split:]

image_data = image_data[:split]
numerical_data = numerical_data[:split]
regression_target = regression_target[:split]
classification_target = classification_target[:split]

model.fit(
    [image_data, numerical_data],
    [regression_target, classification_target],
    # Use your own validation set.
    validation_data=(
        [image_val, numerical_val],
        [regression_val, classification_val],
    ),
    epochs=1,
    batch_size=3,
)

"""
## Customized Search Space
You can customize your search space.
The following figure shows the search space we want to define.

<div class="mermaid">
graph LR
    id1(ImageInput) --> id2(Normalization)
    id2 --> id3(Image Augmentation)
    id3 --> id4(Convolutional)
    id3 --> id5(ResNet V2)
    id4 --> id6(Merge)
    id5 --> id6
    id7(Input) --> id9(DenseBlock)
    id6 --> id10(Merge)
    id9 --> id10
    id10 --> id11(Classification Head)
    id10 --> id12(Regression Head)
</div>
"""

input_node1 = ak.ImageInput()
output_node = ak.Normalization()(input_node1)
output_node = ak.ImageAugmentation()(output_node)
output_node1 = ak.ConvBlock()(output_node)
output_node2 = ak.ResNetBlock(version="v2")(output_node)
output_node1 = ak.Merge()([output_node1, output_node2])

input_node2 = ak.Input()
output_node2 = ak.DenseBlock()(input_node2)

output_node = ak.Merge()([output_node1, output_node2])
output_node1 = ak.ClassificationHead()(output_node)
output_node2 = ak.RegressionHead()(output_node)

auto_model = ak.AutoModel(
    inputs=[input_node1, input_node2],
    outputs=[output_node1, output_node2],
    overwrite=True,
    max_trials=2,
)

image_data = np.random.rand(num_instances, 32, 32, 3).astype(np.float32)
numerical_data = np.random.rand(num_instances, 20).astype(np.float32)
regression_target = np.random.rand(num_instances, 1).astype(np.float32)
classification_target = np.random.randint(5, size=num_instances)

auto_model.fit(
    [image_data, numerical_data],
    [classification_target, regression_target],
    batch_size=3,
    epochs=1,
)

"""
## Data Format
You can refer to the documentation of
[ImageInput](/node/#imageinput-class),
[Input](/node/#input-class),
[TextInput](/node/#textinput-class),
[RegressionHead](/block/#regressionhead-class),
[ClassificationHead](/block/#classificationhead-class),
for the format of different types of data.
You can also refer to the Data Format section of the tutorials of
[Image Classification](/tutorial/image_classification/#data-format),
[Text Classification](/tutorial/text_classification/#data-format),

## Reference
[AutoModel](/auto_model/#automodel-class),
[ImageInput](/node/#imageinput-class),
[Input](/node/#input-class),
[DenseBlock](/block/#denseblock-class),
[RegressionHead](/block/#regressionhead-class),
[ClassificationHead](/block/#classificationhead-class),
[CategoricalToNumerical](/block/#categoricaltonumerical-class).
"""
