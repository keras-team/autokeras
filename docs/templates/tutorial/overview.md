# AutoKeras 1.0 Tutorial

## Supported Tasks

AutoKeras supports several tasks with extremely simple interface.
You can click the links below to see the detailed tutorial for each task.

**Suported Tasks**:

[Image Classification](/tutorial/image_classification)

[Image Regression](/tutorial/image_regression)

[Text Classification](/tutorial/text_classification)

[Text Regression](/tutorial/text_regression)

[Structured Data Classification](/tutorial/structured_data_classification)

[Structured Data Regression](/tutorial/structured_data_regression)

**Coming Soon**: Time Series Forcasting, Object Detection, Image Segmentation.


## Multi-Task and Multi-Modal Data

If you are dealing with multi-task or multi-modal dataset, you can refer to this
[tutorial](/tutorial/multi) for details.


## Customized Model

Follow this [tutorial](/tutorial/customized), to use AutoKeras building blocks to quickly construct your own
model.
With these blocks, you only need to specify the high-level architecture of your
model.
AutoKeras would search for the best detailed configuration for you.
Moreover, you can override the base classes to create your own block.
The following are the links to the documentation of the predefined input nodes and blocks in AutoKeras.

**Nodes**:

[ImageInput](/node/#imageinput-class)

[Input](/node/#input-class)

[StructuredDataInput](/node/#structureddatainput-class)

[TextInput](/node/#textinput-class)

**Blocks**:

[ImageAugmentation](/block/#imageaugmentation-class)

[Normalization](/block/#normalization-class)

[TextToIntSequence](/block/#texttointsequence-class)

[TextToNgramVector](/block/#texttongramvector-class)

[CategoricalToNumerical](/block/#categoricaltonumerical-class)

[ConvBlock](/block/#convblock-class)

[DenseBlock](/block/#denseblock-class)

[Embedding](/block/#embedding-class)

[Merge](/block/#merge-class)

[ResNetBlock](/block/#resnetblock-class)

[RNNBlock](/block/#rnnblock-class)

[SpatialReduction](/block/#spatialreduction-class)

[TemporalReduction](/block/#temporalreduction-class)

[XceptionBlock](/block/#xceptionblock-class)

[ImageBlock](/block/#imageblock-class)

[StructuredDataBlock](/block/#structureddatablock-class)

[TextBlock](/block/#textblock-class)

[ClassificationHead](/block/#classificationhead-class)

[RegressionHead](/block/#regressionhead-class)

## Export Model
You can follow this [tutorial](/tutorial/export) to export the best model.
