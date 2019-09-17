from tensorflow.python.util import nest

from autokeras.hypermodel import block
from autokeras.hypermodel import head as head_module
from autokeras.hypermodel import node
from autokeras.hypermodel import preprocessor


class HyperBlock(block.Block):
    """HyperBlock uses hyperparameters to decide inner Block graph.

    A HyperBlock should be build into connected Blocks instead of individual Keras
    layers. The main purpose of creating the HyperBlock class is for the ease of
    parsing the graph for preprocessors. The graph would be hard to parse if a Block,
    whose inner structure is decided by hyperparameters dynamically, contains both
    preprocessors and Keras layers.

    When the preprocessing layers of Keras are ready to cover all the preprocessors
    in AutoKeras, the preprocessors should be handled by the Keras Model. The
    HyperBlock class should be removed. The subclasses should extend Block class
    directly and the build function should build connected Keras layers instead of
    Blocks.

    # Arguments
        output_shape: Tuple of int(s). Defaults to None. If None, the output shape
            will be inferred from the AutoModel.
        name: String. The name of the block. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, output_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape

    def build(self, hp, inputs=None):
        """Build the HyperModel instead of Keras Model.

        # Arguments
            hp: Hyperparameters. The hyperparameters for building the model.
            inputs: A list of instances of Node.

        # Returns
            An Node instance, the output node of the output Block.
        """
        raise NotImplementedError


class ImageBlock(HyperBlock):
    """HyperBlock for image data.

    The image blocks is a block choosing from ResNetBlock, XceptionBlock, ConvBlock,
    which is controlled by a hyperparameter, 'block_type'.

    # Arguments
        block_type: String. 'resnet', 'xception', 'vanilla'. The type of HyperBlock
            to use. If unspecified, it will be tuned automatically.
        normalize: Boolean. Whether to channel-wise normalize the images.
            If unspecified, it will be tuned automatically.
        augment: Boolean. Whether to do image augmentation. If unspecified,
            it will be tuned automatically.
    """

    def __init__(self,
                 block_type=None,
                 normalize=True,
                 augment=True,
                 seed=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.block_type = block_type
        self.normalize = normalize
        self.augment = augment
        self.seed = seed

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node

        block_type = self.block_type or hp.Choice('block_type',
                                                  ['resnet', 'xception', 'vanilla'],
                                                  default='resnet')

        normalize = self.normalize
        if normalize is None:
            normalize = hp.Choice('normalize', [True, False], default=True)
        augment = self.augment
        if augment is None:
            augment = hp.Choice('augment', [True, False], default=True)
        if normalize:
            output_node = preprocessor.Normalization()(output_node)
        if augment:
            output_node = preprocessor.ImageAugmentation(seed=self.seed)(output_node)
        sub_block_name = self.name + '_' + block_type
        if block_type == 'resnet':
            output_node = block.ResNetBlock(name=sub_block_name)(output_node)
        elif block_type == 'xception':
            output_node = block.XceptionBlock(name=sub_block_name)(output_node)
        elif block_type == 'vanilla':
            output_node = block.ConvBlock(name=sub_block_name).build(output_node)
        return output_node


class TextBlock(HyperBlock):

    def __init__(self, vectorizer=None, pretraining=None, **kwargs):
        super().__init__(**kwargs)
        self.vectorizer = vectorizer
        self.pretraining = pretraining

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node
        vectorizer = self.vectorizer or hp.Choice('vectorizer',
                                                  ['sequence', 'ngram'],
                                                  default='sequence')
        if not isinstance(input_node, node.TextNode):
            raise ValueError('The input_node should be a TextNode.')
        if vectorizer == 'ngram':
            output_node = preprocessor.TextToNgramVector()(output_node)
            output_node = block.DenseBlock()(output_node)
        else:
            output_node = preprocessor.TextToIntSequence()(output_node)
            output_node = block.EmbeddingBlock(
                pretraining=self.pretraining)(output_node)
            output_node = block.ConvBlock(separable=True)(output_node)
        return output_node


class LightGBMClassifierBlock(HyperBlock):
    """Structured data classification with LightGBM.

    It can be used with preprocessors, but not other blocks or heads.

    # Arguments
        metrics: String. The type of the model's metrics. If unspecified,
            it will be 'accuracy' for classification task.
    """

    def __init__(self, metrics=None, **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics
        if self.metrics is None:
            self.metrics = ['accuracy']

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node
        output_node = preprocessor.LightGBMClassifier()(output_node)
        output_node = block.IdentityBlock()(output_node)
        output_node = head_module.EmptyHead(
            loss='categorical_crossentropy',
            metrics=self.metrics,
            output_shape=self.output_shape)(output_node)
        return output_node


class LightGBMRegressorBlock(HyperBlock):
    """Structured data regression with LightGBM.

    It can be used with preprocessors, but not other blocks or heads.

    # Arguments
        metrics: String. The type of the model's metrics. If unspecified,
            it will be 'mean_squared_error' for regression task.
    """

    def __init__(self, metrics=None, **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics
        if self.metrics is None:
            self.metrics = ['mean_squared_error']

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node
        output_node = preprocessor.LightGBMRegressor()(output_node)
        output_node = block.IdentityBlock()(output_node)
        output_node = head_module.EmptyHead(
            loss='mean_squared_error',
            metrics=self.metrics,
            output_shape=self.output_shape)(output_node)
        return output_node


class SupervisedStructuredDataPipelineBlock(HyperBlock):
    """Base class for StructuredDataClassifier(Regressor)Block."""

    def __init__(self,
                 column_types,
                 column_names,
                 feature_engineering=True,
                 module_type=None,
                 head=None,
                 lightgbm_block=None,
                 **kwargs):
        super().__init__()
        self.column_types = column_types
        self.column_names = column_names
        self.feature_engineering = feature_engineering
        self.module_type = module_type
        self.head = head
        self.lightgbm_block = lightgbm_block

    def build_feature_engineering(self, hp, input_node):
        output_node = input_node
        feature_engineering = self.feature_engineering
        if feature_engineering is None:
            # TODO: If False, use plain label encoding.
            feature_engineering = hp.Choice('feature_engineering',
                                            [True],
                                            default=True)
        if feature_engineering:
            output_node = preprocessor.FeatureEngineering(
                column_types=self.column_types,
                column_names=self.column_names)(output_node)
        return output_node

    def build_body(self, hp, input_node):
        module_type = self.module_type or hp.Choice('module_type',
                                                    ['dense', 'lightgbm'],
                                                    default='lightgbm')
        if module_type == 'dense':
            output_node = block.DenseBlock()(input_node)
            self.head.output_shape = self.output_shape
            output_node = self.head(output_node)
        elif module_type == 'lightgbm':
            self.lightgbm_block.output_shape = self.output_shape
            output_node = self.lightgbm_block.build(hp, input_node)
        else:
            raise ValueError('Unsupported module'
                             'type: {module_type}'.format(
                                 module_type=module_type))
        nest.flatten(output_node)[0].shape = self.output_shape
        return output_node

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = self.build_feature_engineering(hp, input_node)
        output_node = self.build_body(hp, output_node)
        return output_node


class StructuredDataBlock(SupervisedStructuredDataPipelineBlock):
    """A block for structured data.

    It searches for whether to use feature engineering. The data is then
    processed with DenseBlock.

    # Arguments
        column_types: A list of strings. The length of the list should be the same
            as the number of columns of the data. The strings in the list are
            specifying the types of the columns. They should either be 'numerical'
            or 'categorical'.
        feature_engineering: Boolean. Whether to use feature engineering for the
            data. If is None, it would be tunable. Defaults to True.
    """

    def __init__(self,
                 column_types,
                 feature_engineering=True,
                 **kwargs):
        super().__init__(column_types=column_types,
                         feature_engineering=feature_engineering,
                         **kwargs)

    def build_body(self, hp, input_node):
        return block.DenseBlock()(input_node)


class StructuredDataClassifierBlock(SupervisedStructuredDataPipelineBlock):
    """A block for structured data classification.

    It cannot be connected with any other block downwards. It searches for whether
    to use feature engineering for the data, and to use DenseBlock and
    ClassificationHead or the LightGBMClassifierBlock.

    # Arguments
        column_types: A list of strings. The length of the list should be the same
            as the number of columns of the data. The strings in the list are
            specifying the types of the columns. They should either be 'numerical'
            or 'categorical'.
        feature_engineering: Boolean. Whether to use feature engineering for the
            data. If is None, it would be tunable. Defaults to True.
        loss: Keras loss function. The loss function for ClassificationHead.
        metrics: A list of Keras metrics. The metrics to use to evaluate the
            classification.
        head: ClassificationHead. The ClassificationHead to use with DenseBlock.
            If unspecified, it would use the default args for the ClassificationHead.
            If specify both head and metrics, the metrics will only be used for
            LightGBM, the head with its metrics will be directly used for DenseBlock.
            If only specified the head, the same metrics will be used for LightGBM.
    """

    def __init__(self,
                 column_types,
                 column_names,
                 feature_engineering=True,
                 loss=None,
                 metrics=None,
                 head=None,
                 **kwargs):
        self.loss = loss
        self.metrics = metrics
        self.head = head
        if self.metrics is None and isinstance(self.head, head_module.Head):
            self.metrics = head.metrics
        if self.head is None:
            self.head = head_module.ClassificationHead(loss=self.loss,
                                                       metrics=self.metrics)
        super().__init__(
            column_types=column_types,
            column_names=column_names,
            feature_engineering=feature_engineering,
            head=self.head,
            lightgbm_block=LightGBMClassifierBlock(metrics=self.metrics))


class StructuredDataRegressorBlock(SupervisedStructuredDataPipelineBlock):
    """A block for structured data regression.

    It cannot be connected with any other block downwards. It searches for whether
    to use feature engineering for the data, and to use DenseBlock and
    RegressionHead or the LightGBMRegressorBlock.

    # Arguments
        column_types: A list of strings. The length of the list should be the same
            as the number of columns of the data. The strings in the list are
            specifying the types of the columns. They should either be 'numerical'
            or 'categorical'.
        feature_engineering: Boolean. Whether to use feature engineering for the
            data. If is None, it would be tunable. Defaults to True.
        loss: Keras loss function. The loss function for RegressionHead.
        metrics: A list of Keras metrics. The metrics to use to evaluate the
            regression.
        head: RegressionHead. The RegressionHead to use with DenseBlock.
            If unspecified, it would use the default args for the RegressionHead.
            If specify both head and metrics, the metrics will only be used for
            LightGBM, the head with its metrics will be directly used for DenseBlock.
            If only specified the head, the same metrics will be used for LightGBM.
    """

    def __init__(self,
                 column_types,
                 column_names,
                 feature_engineering=True,
                 loss=None,
                 metrics=None,
                 head=None,
                 **kwargs):
        self.loss = loss
        self.metrics = metrics
        self.head = head
        if self.metrics is None and isinstance(self.head, head_module.Head):
            self.metrics = head.metrics
        if self.head is None:
            self.head = head_module.RegressionHead(loss=self.loss,
                                                   metrics=self.metrics),
        super().__init__(
            column_types=column_types,
            column_names=column_names,
            feature_engineering=feature_engineering,
            head=self.head,
            lightgbm_block=LightGBMRegressorBlock(metrics=self.metrics))


class TimeSeriesBlock(HyperBlock):

    def build(self, hp, inputs=None):
        raise NotImplementedError


class GeneralBlock(HyperBlock):
    """A general neural network block when the input type is unknown.

    When the input type is unknown. The GeneralBlock would search in a large space
    for a good model.

    # Arguments
        name: String.
    """

    def build(self, hp, inputs=None):
        raise NotImplementedError
