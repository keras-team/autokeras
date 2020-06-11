from tensorflow.python.util import nest

from autokeras.blocks import basic
from autokeras.blocks import preprocessing
from autokeras.blocks import reduction
from autokeras.engine import block as block_module


class ImageBlock(block_module.Block):
    """Block for image data.

    The image blocks is a block choosing from ResNetBlock, XceptionBlock, ConvBlock,
    which is controlled by a hyperparameter, 'block_type'.

    # Arguments
        block_type: String. 'resnet', 'xception', 'vanilla'. The type of Block
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
                 **kwargs):
        super().__init__(**kwargs)
        self.block_type = block_type
        self.normalize = normalize
        self.augment = augment

    def get_config(self):
        config = super().get_config()
        config.update({'block_type': self.block_type,
                       'normalize': self.normalize,
                       'augment': self.augment})
        return config

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node

        block_type = self.block_type or hp.Choice('block_type',
                                                  ['resnet', 'xception', 'vanilla'],
                                                  default='vanilla')

        normalize = self.normalize
        if normalize is None:
            normalize = hp.Boolean('normalize', default=False)
        augment = self.augment
        if augment is None:
            augment = hp.Boolean('augment', default=False)
        if normalize:
            output_node = preprocessing.Normalization().build(hp, output_node)
        if augment:
            output_node = preprocessing.ImageAugmentation().build(hp, output_node)
        if block_type == 'resnet':
            output_node = basic.ResNetBlock().build(hp, output_node)
        elif block_type == 'xception':
            output_node = basic.XceptionBlock().build(hp, output_node)
        elif block_type == 'vanilla':
            output_node = basic.ConvBlock().build(hp, output_node)
        return output_node


class TextBlock(block_module.Block):
    """Block for text data.

    # Arguments
        block_type: String. 'vanilla', 'transformer', and 'ngram'. The type of Block
            to use. 'vanilla' and 'transformer' use a TextToIntSequence vectorizer,
            whereas 'ngram' uses TextToNgramVector. If unspecified, it will be tuned
            automatically.
        max_tokens: Int. The maximum size of the vocabulary.
            If left unspecified, it will be tuned automatically.
        pretraining: String. 'random' (use random weights instead any pretrained
            model), 'glove', 'fasttext' or 'word2vec'. Use pretrained word embedding.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 block_type=None,
                 max_tokens=None,
                 pretraining=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.block_type = block_type
        self.max_tokens = max_tokens
        self.pretraining = pretraining

    def get_config(self):
        config = super().get_config()
        config.update({
            'block_type': self.block_type,
            'max_tokens': self.max_tokens,
            'pretraining': self.pretraining})
        return config

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node
        block_type = self.block_type or hp.Choice('block_type',
                                                  ['vanilla',
                                                   'transformer',
                                                   'ngram'],
                                                  default='vanilla')
        max_tokens = self.max_tokens or hp.Choice('max_tokens',
                                                  [500, 5000, 20000],
                                                  default=5000)
        if block_type == 'ngram':
            output_node = preprocessing.TextToNgramVector(
                max_tokens=max_tokens).build(hp, output_node)
            output_node = basic.DenseBlock().build(hp, output_node)
        else:
            output_node = preprocessing.TextToIntSequence(
                max_tokens=max_tokens).build(hp, output_node)
            if block_type == 'transformer':
                output_node = basic.Transformer(max_features=max_tokens + 1,
                                                pretraining=self.pretraining
                                                ).build(hp, output_node)
            else:
                output_node = basic.Embedding(
                    max_features=max_tokens + 1,
                    pretraining=self.pretraining).build(hp, output_node)
                output_node = basic.ConvBlock().build(hp, output_node)
            output_node = reduction.SpatialReduction().build(hp, output_node)
            output_node = basic.DenseBlock().build(hp, output_node)
        return output_node


class StructuredDataBlock(block_module.Block):
    """Block for structured data.

    # Arguments
        categorical_encoding: Boolean. Whether to use the CategoricalToNumerical to
            encode the categorical features to numerical features. Defaults to True.
            If specified as None, it will be tuned automatically.
        seed: Int. Random seed.
    """

    def __init__(self,
                 categorical_encoding=True,
                 seed=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.categorical_encoding = categorical_encoding
        self.seed = seed
        self.column_types = None
        self.column_names = None

    @classmethod
    def from_config(cls, config):
        column_types = config.pop('column_types')
        column_names = config.pop('column_names')
        instance = cls(**config)
        instance.column_types = column_types
        instance.column_names = column_names
        return instance

    def get_config(self):
        config = super().get_config()
        config.update({'categorical_encoding': self.categorical_encoding,
                       'seed': self.seed,
                       'column_types': self.column_types,
                       'column_names': self.column_names})
        return config

    def build_categorical_encoding(self, hp, input_node):
        output_node = input_node
        categorical_encoding = self.categorical_encoding
        if categorical_encoding is None:
            categorical_encoding = hp.Choice('categorical_encoding',
                                             [True, False],
                                             default=True)
        if categorical_encoding:
            block = preprocessing.CategoricalToNumerical()
            block.column_types = self.column_types
            block.column_names = self.column_names
            output_node = block.build(hp, output_node)
        return output_node

    def build_body(self, hp, input_node):
        output_node = basic.DenseBlock().build(hp, input_node)
        return output_node

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = self.build_categorical_encoding(hp, input_node)
        output_node = self.build_body(hp, output_node)
        return output_node


class TimeseriesBlock(block_module.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node
        output_node = basic.RNNBlock().build(hp, output_node)
        return output_node


class GeneralBlock(block_module.Block):
    """A general neural network block when the input type is unknown.

    When the input type is unknown. The GeneralBlock would search in a large space
    for a good model.

    # Arguments
        name: String.
    """

    def build(self, hp, inputs=None):
        raise NotImplementedError
