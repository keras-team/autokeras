In this tutorial, we show how to customize your search space with
[AutoModel](/auto_model/#automodel-class) and how to implement your own block as search space.
This API is mainly for advanced users who already know what their model should look like.

## Customized Search Space
First, let us see how we can build the following neural network using the building blocks in AutoKeras.

<div class="mermaid">
graph LR
    id1(ImageInput) --> id2(Normalization)
    id2 --> id3(Image Augmentation)
    id3 --> id4(Convolutional)
    id3 --> id5(ResNet V2)
    id4 --> id6(Merge)
    id5 --> id6
    id6 --> id7(Classification Head)
</div>

We can make use of the [AutoModel](/auto_model/#automodel-class) API in AutoKeras to implemented as follows.
The usage is the same as the [Keras functional API](https://www.tensorflow.org/guide/keras/functional).

```python
import autokeras as ak

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation()(output_node)
output_node1 = ak.ConvBlock()(output_node)
output_node2 = ak.ResNetBlock(version='v2')(output_node)
output_node = ak.Merge()([output_node1, output_node2])

auto_model = ak.AutoModel(
    inputs=input_node, 
    outputs=output_node,
    max_trials=10)
```

Whild building the model, the blocks used need to follow this topology:
`Preprocessor` -> `Block` -> `Head`. `Normalization` and `ImageAugmentation` are `Preprocessor`s.
`ClassificationHead` is `Head`. The rest are `Block`s.

In the code above, we use `ak.ResNetBlock(version='v2')` to specify the version of ResNet to use.
There are many other arguments to specify for each building block.
For most of the arguments, if not specified, they would be tuned automatically.
Please refer to the documentation links at the bottom of the page for more details.

Then, we prepare some data to run the model.

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
print(y_train[:3]) # array([7, 2, 1], dtype=uint8)

# Feed the AutoModel with training data.
auto_model.fit(x_train, y_train)
# Predict with the best model.
predicted_y = auto_model.predict(x_test)
# Evaluate the best model with testing data.
print(auto_model.evaluate(x_test, y_test))
```

For multiple input nodes and multiple heads search space, you can refer to [this section](/tutorial/multi/#customized-search-space).

## Validation Data
If you would like to provide your own validation data or change the ratio of the validation data, please refer to
the Validation Data section of the tutorials of
[Image Classification](/tutorial/image_classification/#validation-data),
[Text Classification](/tutorial/text_classification/#validation-data),
[Structured Data Classification](/tutorial/structured_data_classification/#validation-data),
[Multi-task and Multiple Validation](/tutorial/multi/#validation-data).

## Data Format
You can refer to the documentation of
[ImageInput](/node/#imageinput-class),
[StructuredDataInput](/node/#structureddatainput-class),
[TextInput](/node/#textinput-class),
[RegressionHead](/head/#regressionhead-class),
[ClassificationHead](/head/#classificationhead-class),
for the format of different types of data.
You can also refer to the Data Format section of the tutorials of
[Image Classification](/tutorial/image_classification/#data-format),
[Text Classification](/tutorial/text_classification/#data-format),
[Structured Data Classification](/tutorial/structured_data_classification/#data-format).

## Implement New Block

We are still working on this tutorial. Thank you for your patience!

## Reference

**Nodes**:
[ImageInput](/node/#imageinput-class),
[Input](/node/#input-class),
[StructuredDataInput](/node/#structureddatainput-class),
[TextInput](/node/#textinput-class).

**Preprocessors**:
[FeatureEngineering](/preprocessor/#featureengineering-class),
[ImageAugmentation](/preprocessor/#imageaugmentation-class),
[LightGBM](/preprocessor/#lightgbm-class),
[Normalization](/preprocessor/#normalization-class),
[TextToIntSequence](/preprocessor/#texttointsequence-class),
[TextToNgramVector](/preprocessor/#texttongramvector-class).

**Blocks**:
[ConvBlock](/block/#convblock-class),
[DenseBlock](/block/#denseblock-class),
[EmbeddingBlock](/block/#embeddingblock-class),
[Merge](/block/#merge-class),
[ResNetBlock](/block/#resnetblock-class),
[RNNBlock](/block/#rnnblock-class),
[SpatialReduction](/block/#spatialreduction-class),
[TemporalReduction](/block/#temporalreduction-class),
[XceptionBlock](/block/#xceptionblock-class),
[ImageBlock](/block/#imageblock-class),
[StructuredDataBlock](/block/#structureddatablock-class),
[TextBlock](/block/#textblock-class).

