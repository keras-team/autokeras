# AutoKeras 1.0 Tutorial

## Supported Tasks

AutoKeras supports several tasks with extremely simple interface.
You can click the links below to see the detailed tutorial for each task.

[Image Classification](/tutorial/image_classification)

[Image Regression](/tutorial/image_regression)

[Text Classification](/tutorial/text_classification)

[Text Regression](/tutorial/text_regression)

[Structured Data Classification](/tutorial/structured_data_classification)

[Structured Data Regression](/tutorial/structured_data_regression)


Coming Soon: Time Series Forcasting, Object Detection, Image Segmentation.


## Multi-Task and Multi-Modal Data

If you are dealing with multi-task or multi-modal dataset, you can refer to [this
tutorial](/tutorial/multi) for details.


## Customized Model

AutoKeras also provide many building blocks for you to quickly construct your own
model.
With these blocks, you only need to specify the high-level architecture of your
model.
AutoKeras would search for the best detailed configuration for you.

### Nodes
[ImageInput](/node/#imageinput-class)

[Input](/node/#input-class)

[StructuredDataInput](/node/#structureddatainput-class)

[TextInput](/node/#textinput-class)

### Preprocessors
[FeatureEngineering](/preprocessor/#featureengineering-class)

[ImageAugmentation](/preprocessor/#imageaugmentation-class)

[LightGBMBlock](/preprocessor/#lightgbmblock-class)

[Normalization](/preprocessor/#normalization-class)

[TextToIntSequence](/preprocessor/#texttointsequence-class)

[TextToNgramVector](/preprocessor/#texttongramvector-class)

### Blocks
ConvBlock
DenseBlock
EmbeddingBlock
Merge
ResNetBlock
RNNBlock
SpatialReduction
TemporalReduction
XceptionBlock
ImageBlock
StructuredDataBlock
TextBlock

### Heads
ClassificationHead
RegressionHead
