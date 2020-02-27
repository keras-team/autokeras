from tensorflow.python.util import nest

from autokeras.engine import block as block_module
from autokeras.hypermodels import basic
from autokeras.hypermodels import preprocessing
from autokeras.hypermodels import reduction


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
        vectorizer: String. 'sequence' or 'ngram'. If it is 'sequence',
            TextToIntSequence will be used. If it is 'ngram', TextToNgramVector will
            be used. If unspecified, it will be tuned automatically.
        pretraining: String. 'random' (use random weights instead any pretrained
            model), 'glove', 'fasttext' or 'word2vec'. Use pretrained word embedding.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, vectorizer=None, pretraining=None, **kwargs):
        super().__init__(**kwargs)
        self.vectorizer = vectorizer
        self.pretraining = pretraining

    def get_config(self):
        config = super().get_config()
        config.update({'vectorizer': self.vectorizer,
                       'pretraining': self.pretraining})
        return config

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node
        vectorizer = self.vectorizer or hp.Choice('vectorizer',
                                                  ['sequence', 'ngram'],
                                                  default='sequence')
        if vectorizer == 'ngram':
            output_node = preprocessing.TextToNgramVector().build(hp, output_node)
            output_node = basic.DenseBlock().build(hp, output_node)
        else:
            output_node = preprocessing.TextToIntSequence().build(hp, output_node)
            output_node = basic.Embedding(
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

    def get_config(self):
        config = super().get_config()
        config.update({'categorical_encoding': self.categorical_encoding,
                       'seed': self.seed})
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


class TimeSeriesBlock(block_module.Block):

    def build(self, hp, inputs=None):
        raise NotImplementedError


class GeneralBlock(block_module.Block):
    """A general neural network block when the input type is unknown.

    When the input type is unknown. The GeneralBlock would search in a large space
    for a good model.

    # Arguments
        name: String.
    """

    def build(self, hp, inputs=None):
        raise NotImplementedError
