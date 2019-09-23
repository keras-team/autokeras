from tensorflow.python.util import nest

from autokeras import utils
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
    """Block for image data.

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
    """Block for text data.

    # Arguments
        vectorizer: String. 'sequence' or 'ngram'. If it is 'sequence',
            TextToIntSequence will be used. If it is 'ngram', TextToNgramVector will
            be used. If unspecified, it will be tuned automatically.
        pretraining: Boolean. Whether to use pretraining weights in the N-gram
            vectorizer. If unspecified, it will be tuned automatically.
    """

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


class StructuredDataBlock(HyperBlock):
    """Block for structured data.

    # Arguments
        feature_engineering: Boolean. Whether to use feature engineering block.
            Defaults to True. If specified as None, it will be tuned automatically.
        module_type: String. 'dense' or 'lightgbm'. If it is 'dense', DenseBlock
            will be used. If it is 'lightgbm', LightGBMBlock will be used. If
            unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 feature_engineering=True,
                 module_type=None,
                 **kwargs):
        super().__init__()
        self.feature_engineering = feature_engineering
        self.module_type = module_type
        self.heads = None

    def build_feature_engineering(self, hp, input_node):
        output_node = input_node
        feature_engineering = self.feature_engineering
        if feature_engineering is None:
            # TODO: If False, use plain label encoding.
            feature_engineering = hp.Choice('feature_engineering',
                                            [True],
                                            default=True)
        if feature_engineering:
            output_node = preprocessor.FeatureEngineering()(output_node)
        return output_node

    def build_body(self, hp, input_node):
        if len(self.heads) > 1:
            module_type_choices = ['dense']
        else:
            module_type_choices = ['lightgbm', 'dense']
        module_type = self.module_type or hp.Choice('module_type',
                                                    module_type_choices,
                                                    default=module_type_choices[0])
        if module_type == 'dense':
            output_node = block.DenseBlock()(input_node)
        elif module_type == 'lightgbm':
            output_node = preprocessor.LightGBMBlock()(input_node)
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

    def compile(self):
        self.heads = head_module.fetch_heads(self)


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
