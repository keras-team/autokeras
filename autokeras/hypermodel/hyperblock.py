from tensorflow.python.util import nest

from autokeras.hypermodel import base
from autokeras.hypermodel import block as block_module
from autokeras.hypermodel import node as node_module
from autokeras.hypermodel import preprocessor as preprocessor_module


class ImageBlock(base.HyperBlock):
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
                 normalize=None,
                 augment=None,
                 seed=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.block_type = block_type
        self.normalize = normalize
        self.augment = augment
        self.seed = seed

    def get_state(self):
        config = super().get_state()
        config.update({'block_type': self.block_type,
                       'normalize': self.normalize,
                       'augment': self.augment,
                       'seed': self.seed})
        return config

    def set_state(self, state):
        super().set_state(state)
        self.block_type = state.get('block_type')
        self.normalize = state.get('normalize')
        self.augment = state.get('augment')
        self.seed = state.get('seed')

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node

        block_type = self.block_type or hp.Choice('block_type',
                                                  ['resnet', 'xception', 'vanilla'],
                                                  default='vanilla')

        normalize = self.normalize
        if normalize is None:
            normalize = hp.Choice('normalize', [True, False], default=True)
        augment = self.augment
        if augment is None:
            augment = hp.Choice('augment', [True, False], default=False)
        if normalize:
            output_node = preprocessor_module.Normalization()(output_node)
        if augment:
            output_node = preprocessor_module.ImageAugmentation(
                seed=self.seed)(output_node)
        sub_block_name = self.name + '_' + block_type
        if block_type == 'resnet':
            output_node = block_module.ResNetBlock(name=sub_block_name)(output_node)
        elif block_type == 'xception':
            output_node = block_module.XceptionBlock(
                name=sub_block_name)(output_node)
        elif block_type == 'vanilla':
            output_node = block_module.ConvBlock(name=sub_block_name)(output_node)
        return output_node


class TextBlock(base.HyperBlock):
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

    def get_state(self):
        state = super().get_state()
        state.update({'vectorizer': self.vectorizer,
                      'pretraining': self.pretraining})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.vectorizer = state['vectorizer']
        self.pretraining = state['pretraining']

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node
        vectorizer = self.vectorizer or hp.Choice('vectorizer',
                                                  ['sequence', 'ngram'],
                                                  default='sequence')
        if not isinstance(input_node, node_module.TextNode):
            raise ValueError('The input_node should be a TextNode.')
        if vectorizer == 'ngram':
            output_node = preprocessor_module.TextToNgramVector()(output_node)
            output_node = block_module.DenseBlock()(output_node)
        else:
            output_node = preprocessor_module.TextToIntSequence()(output_node)
            output_node = block_module.EmbeddingBlock(
                pretraining=self.pretraining)(output_node)
            output_node = block_module.ConvBlock(separable=True)(output_node)
            output_node = block_module.SpatialReduction()(output_node)
            output_node = block_module.DenseBlock()(output_node)
        return output_node


class StructuredDataBlock(base.HyperBlock):
    """Block for structured data.

    # Arguments
        feature_engineering: Boolean. Whether to use feature engineering block.
            Defaults to True. If specified as None, it will be tuned automatically.
        module_type: String. 'dense' or 'lightgbm'. If it is 'dense', DenseBlock
            will be used. If it is 'lightgbm', LightGBMBlock will be used. If
            unspecified, it will be tuned automatically.
        seed: Int. Random seed.
    """

    def __init__(self,
                 feature_engineering=True,
                 module_type=None,
                 seed=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_engineering = feature_engineering
        self.module_type = module_type
        self.num_heads = None
        self.seed = seed

    def get_state(self):
        config = super().get_state()
        config.update({'feature_engineering': self.feature_engineering,
                       'module_type': self.module_type,
                       'num_heads': self.num_heads,
                       'seed': self.seed})
        return config

    def set_state(self, state):
        super().set_state(state)
        self.feature_engineering = state.get('feature_engineering')
        self.module_type = state.get('module_type')
        self.num_heads = state.get('num_heads')
        self.seed = state.get('seed')

    def build_feature_engineering(self, hp, input_node):
        output_node = input_node
        feature_engineering = self.feature_engineering
        if feature_engineering is None:
            # TODO: If False, use plain label encoding.
            feature_engineering = hp.Choice('feature_engineering',
                                            [True],
                                            default=True)
        if feature_engineering:
            output_node = preprocessor_module.FeatureEngineering()(output_node)
        return output_node

    def build_body(self, hp, input_node):
        if self.num_heads > 1:
            module_type_choices = ['dense']
        else:
            module_type_choices = ['lightgbm', 'dense']
        module_type = self.module_type or hp.Choice('module_type',
                                                    module_type_choices,
                                                    default=module_type_choices[0])
        if module_type == 'dense':
            output_node = block_module.DenseBlock()(input_node)
        elif module_type == 'lightgbm':
            output_node = preprocessor_module.LightGBMBlock(
                seed=self.seed)(input_node)
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


class TimeSeriesBlock(base.HyperBlock):

    def build(self, hp, inputs=None):
        raise NotImplementedError


class GeneralBlock(base.HyperBlock):
    """A general neural network block when the input type is unknown.

    When the input type is unknown. The GeneralBlock would search in a large space
    for a good model.

    # Arguments
        name: String.
    """

    def build(self, hp, inputs=None):
        raise NotImplementedError
