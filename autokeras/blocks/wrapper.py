import numpy as np
from kerastuner.applications import xception
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_source_inputs
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
        max_tokens: Int. The maximum size of the vocabulary.
            If left unspecified, it will be tuned automatically.
        vectorizer: String. 'sequence' or 'ngram'. If it is 'sequence',
            TextToIntSequence will be used. If it is 'ngram', TextToNgramVector will
            be used. If unspecified, it will be tuned automatically.
        pretraining: String. 'random' (use random weights instead any pretrained
            model), 'glove', 'fasttext' or 'word2vec'. Use pretrained word embedding.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 max_tokens=None,
                 vectorizer=None,
                 pretraining=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.vectorizer = vectorizer
        self.pretraining = pretraining

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_tokens': self.max_tokens,
            'vectorizer': self.vectorizer,
            'pretraining': self.pretraining})
        return config

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node
        vectorizer = self.vectorizer or hp.Choice('vectorizer',
                                                  ['sequence', 'ngram'],
                                                  default='sequence')
        max_tokens = self.max_tokens or hp.Choice('max_tokens',
                                                  [500, 5000, 20000],
                                                  default=5000)
        if vectorizer == 'ngram':
            output_node = preprocessing.TextToNgramVector(
                max_tokens=max_tokens).build(hp, output_node)
            output_node = basic.DenseBlock().build(hp, output_node)
        else:
            output_node = preprocessing.TextToIntSequence(
                max_tokens=max_tokens).build(hp, output_node)
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


class SegmentationBlock(block_module.Block):
    """Block for image semantic segmentation.

    The image block is a block choosing from ResNetBlock, XceptionBlock, ConvBlock,
    which is controlled by a hyperparameter, 'block_type' from the paper
    https://arxiv.org/pdf/1606.00915.pdf.

    This image block is the task of semantic segmentation by applying g the
    ‘atrous convolution’ with upsampled filters for dense feature extraction.
    Then further extend it to atrous spatial pyramid pooling, which encodes
    objects as well as image context at multiple scales.
    To produce semantically accurate predictions and detailed segmentation maps
    along object boundaries, we also combine ideas from deep
    convolutional neural networks and fully-connected conditional random fields.

    # Arguments
        block_type: String. 'resnet', 'xception', 'vanilla'. The type of Block
            to use. If unspecified, it will be tuned automatically.
        classes: Int. Number of desired classes. If classes != 21,
            last layer is initialized randomly.
        backbone: String. Backbone to use. one of {'xception','mobilenetv2'}
        os: Int. Determines input_shape/feature_extractor_output ratio.
            One of {8,16}. Used only for xception backbone which means
            that the output size is the the encoder is 1/os of the
            original image size.
        alpha: Float. Controls the width of the MobileNetV2 network.
            This is known as the width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                        of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                        of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                        are used at each layer.
            Used only for mobilenetv2 backbone
    """

    def __init__(self,
                 block_type=None,
                 os=None,
                 alpha=None,
                 classes=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.block_type = block_type
        self.os = os
        self.alpha = alpha
        self.classes = classes

    def get_config(self):
        config = super().get_config()
        config.update({'block_type': self.block_type,
                       'os': self.os,
                       'alpha': self.alpha,
                       'classes': self.classes})
        return config

    def _xception_block(self, inputs, depth_list, prefix, skip_connection_type,
                        stride, rate=1, depth_activation=False, return_skip=False):
        """Basic building block of modified Xception network.

        # Arguments
            x: Numpy.ndarray or tensorflow.Dataset. Input tensor.
            depth_list: Int. Number of filters in each SepConv layer.
                len(depth_list) == 3.
            prefix: String. Prefix before name.
            skip_connection_type: String. One of {'conv','sum','none'}
            stride: Int. Stride at depthwise convolution.
            rate: Int. Atrous rate for depthwise convolution.
            depth_activation: String. Activation function. One of {'relu', 'selu'}
            return_skip: Boolean. Flag to return additional tensor after
                2 SepConvs for decoder
                """
        residual = inputs
        for i in range(3):
            residual = xception.sep_conv(residual, depth_list[i],
                                         kernel_size=3,
                                         activation=depth_activation)
            if i == 1:
                skip = residual
        if skip_connection_type == 'conv':
            shortcut = xception.conv(inputs, depth_list[-1],
                                     kernel_size=1, activation='relu',
                                     strides=stride)
            shortcut = layers.BatchNormalization(
                name=prefix + '_shortcut_BN')(shortcut)
            outputs = self.layers.add([residual, shortcut])
        elif skip_connection_type == 'sum':
            outputs = self.layers.add([residual, inputs])
        elif skip_connection_type == 'none':
            outputs = residual
        if return_skip:
            return outputs, skip
        else:
            return outputs

    def bulid(self, hp, inputs=None):
        """
        # Arguments
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: shape of input image. format HxWxC
                PASCAL VOC model was trained on (512,512,3) images
        # Returns
            A Keras model instance.
        # Raises
            RuntimeError: If attempting to run this model with a
                backend that does not support separable convolutions.
            ValueError: in case of invalid argument for `weights` or `backbone`
        """

        block_type = self.block_type or hp.Choice('block_type',
                                                  ['mobilenetv2', 'xception'],
                                                  default='xception')

        if K.backend() != 'tensorflow':
            raise RuntimeError('The Deeplabv3+ model is only available with '
                               'the TensorFlow backend.')

        if not (block_type in {'xception', 'mobilenetv2'}):
            raise ValueError('The `backbone` argument should be either '
                             '`xception`  or `mobilenetv2` ')

        if inputs is None:
            img_input = layers.Input(inputs.shape.as_list()[:3])
        else:
            if not K.is_keras_tensor(inputs):
                img_input = layers.Input(
                    tensor=inputs, shape=inputs.shape.as_list()[:3])
            else:
                img_input = inputs

        batches_input = layers.Lambda(lambda x: x / 127.5 - 1)(img_input)
        input_shape = inputs.shape.as_list()[:3]
        if block_type == 'xception':
            if self.os == 8:
                entry_block3_stride = 1
                middle_block_rate = 2  # ! Not mentioned in paper, but required
                exit_block_rates = (2, 4)
            else:
                entry_block3_stride = 2
                middle_block_rate = 1
                exit_block_rates = (1, 2)
            x = basic.ConvBlock(32, kernel_size=3, padding='same',
                                use_bias=False, stride=2,
                                name='entry_flow_conv1_1')\
                .build(hp, inputs=batches_input)

            x = layers.BatchNormalization(name='entry_flow_conv1_1_BN')(x)
            x = layers.Activation('relu')(x)

            x = xception.conv(x, 64, kernel_size=3, activation='relu',
                              strides=1)

            x = layers.BatchNormalization(name='entry_flow_conv1_2_BN')(x)
            x = layers.Activation('relu')(x)

            x = self._xception_block(x, [128, 128, 128], 'entry_flow_block1',
                                     skip_connection_type='conv', stride=2,
                                     depth_activation=False)
            x, skip1 = self._xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                            skip_connection_type='conv', stride=2,
                                            depth_activation=False, return_skip=True)

            x = self._xception_block(x, [728, 728, 728], 'entry_flow_block3',
                                     skip_connection_type='conv',
                                     stride=entry_block3_stride,
                                     depth_activation=False)
            for i in range(16):
                x = self._xception_block(x, [728, 728, 728],
                                         'middle_flow_unit_{}'.format(i + 1),
                                         skip_connection_type='sum', stride=1,
                                         rate=middle_block_rate,
                                         depth_activation=False)

            x = self._xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                                     skip_connection_type='conv',
                                     stride=1,
                                     rate=exit_block_rates[0],
                                     depth_activation=False)
            x = self._xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                                     skip_connection_type='none',
                                     stride=1,
                                     rate=exit_block_rates[1],
                                     depth_activation=True)

        else:
            self.os = 8
            first_block_filters = max(8, int(32 * self.alpha + 8 / 2) // 8 * 8)
            # Make sure that round down does not go down by more than 10%.
            if first_block_filters < 0.9 * 32 * self.alpha:
                first_block_filters += 8
            x = basic.ConvBlock(first_block_filters, kernel_size=3,
                                padding='same',
                                use_bias=False, stride=2,
                                name='Conv').build(hp, inputs=batches_input)
            x = layers.BatchNormalization(
                epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)

            x = layers.Lambda(lambda x: relu(x, max_value=6.))(x)

            x = xception._inverted_res_block(x, filters=16, alpha=self.alpha,
                                             stride=1,
                                             expansion=1, block_id=0,
                                             skip_connection=False)

            x = xception._inverted_res_block(x, filters=24, alpha=self.alpha,
                                             stride=2,
                                             expansion=6, block_id=1,
                                             skip_connection=False)
            x = xception._inverted_res_block(x, filters=24, alpha=self.alpha,
                                             stride=1,
                                             expansion=6, block_id=2,
                                             skip_connection=True)

            x = xception._inverted_res_block(x, filters=32, alpha=self.alpha,
                                             stride=2,
                                             expansion=6, block_id=3,
                                             skip_connection=False)
            x = xception._inverted_res_block(x, filters=32, alpha=self.alpha,
                                             stride=1,
                                             expansion=6, block_id=4,
                                             skip_connection=True)
            x = xception._inverted_res_block(x, filters=32, alpha=self.alpha,
                                             stride=1,
                                             expansion=6, block_id=5,
                                             skip_connection=True)

            # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
            x = xception._inverted_res_block(x, filters=64,
                                             alpha=self.alpha, stride=1,  # 1!
                                             expansion=6, block_id=6,
                                             skip_connection=False)
            x = xception._inverted_res_block(x, filters=64, alpha=self.alpha,
                                             stride=1, rate=2,
                                             expansion=6, block_id=7,
                                             skip_connection=True)
            x = xception._inverted_res_block(x, filters=64, alpha=self.alpha,
                                             stride=1, rate=2,
                                             expansion=6, block_id=8,
                                             skip_connection=True)
            x = xception._inverted_res_block(x, filters=64, alpha=self.alpha,
                                             stride=1, rate=2,
                                             expansion=6, block_id=9,
                                             skip_connection=True)

            x = xception._inverted_res_block(x, filters=96, alpha=self.alpha,
                                             stride=1, rate=2,
                                             expansion=6, block_id=10,
                                             skip_connection=False)
            x = xception._inverted_res_block(x, filters=96, alpha=self.alpha,
                                             stride=1, rate=2,
                                             expansion=6, block_id=11,
                                             skip_connection=True)
            x = xception._inverted_res_block(x, filters=96, alpha=self.alpha,
                                             stride=1, rate=2,
                                             expansion=6, block_id=12,
                                             skip_connection=True)

            x = xception._inverted_res_block(x, filters=160, alpha=self.alpha,
                                             stride=1, rate=2,
                                             expansion=6, block_id=13,
                                             skip_connection=False)
            x = xception._inverted_res_block(x, filters=160, alpha=self.alpha,
                                             stride=1, rate=4,
                                             expansion=6, block_id=14,
                                             skip_connection=True)
            x = xception._inverted_res_block(x, filters=160, alpha=self.alpha,
                                             stride=1, rate=4,
                                             expansion=6, block_id=15,
                                             skip_connection=True)

            x = xception._inverted_res_block(x, filters=320, alpha=self.alpha,
                                             stride=1, rate=4,
                                             expansion=6, block_id=16,
                                             skip_connection=False)

        # end of feature extractor

        # branching for Atrous Spatial Pyramid Pooling

        # Image Feature branch
        # out_shape = int(np.ceil(input_shape[0] / OS))
        b4 = layers.AveragePooling2D(pool_size=(
            int(np.ceil((512, 512, 3)[0] / self.os)),
            int(np.ceil((512, 512, 3)[1] / self.os))))(x)
        x = basic.ConvBlock(256, kernel_size=1, padding='same',
                            use_bias=False,
                            name='image_pooling').build(hp, inputs=b4)
        b4 = layers.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = layers.Activation('relu')(b4)

        b4 = layers.Lambda(lambda x: K.tf.image.resize_bilinear(x, size=(
            int(np.ceil(input_shape[0] / self.os)),
            int(np.ceil(input_shape[1] / self.os)))))(b4)

        # simple 1x1
        b0 = basic.ConvBlock(256, kernel_size=1, padding='same',
                             use_bias=False, name='aspp0')\
            .build(hp, inputs=x)
        b0 = layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = layers.Activation('relu', name='aspp0_activation')(b0)

        # there are only 2 branches in mobilenetV2. not sure why
        if block_type == 'xception':
            # rate = 6 (12)
            b1 = xception.sep_conv(x, 256,
                                   kernel_size=3,
                                   activation='relu')
            # rate = 12 (24)
            b2 = xception.sep_conv(x, 256,
                                   kernel_size=3,
                                   activation='relu')
            # rate = 18 (36)
            b3 = xception.sep_conv(x, 256,
                                   kernel_size=3,
                                   activation='relu')

            # concatenate ASPP branches & project
            x = layers.Concatenate()([b4, b0, b1, b2, b3])
        else:
            x = layers.Concatenate()([b4, b0])

        x = basic.ConvBlock(256, kernel_size=1, padding='same',
                            use_bias=False,
                            name='concat_projection').build(hp, inputs=x)
        x = layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)

        # DeepLab v.3+ decoder

        if block_type == 'xception':
            # Feature projection
            # x4 (x2) block

            x = layers.Lambda(lambda x: K.tf.image.resize_bilinear(x, size=(
                int(np.ceil(input_shape[0] / 4)),
                int(np.ceil(input_shape[1] / 4)))))(x)

            dec_skip1 = basic.ConvBlock(48, kernel_size=1, padding='same',
                                        use_bias=False,
                                        name='feature_projection0')\
                .build(hp, inputs=skip1)
            dec_skip1 = layers.BatchNormalization(
                name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
            dec_skip1 = layers.Activation('relu')(dec_skip1)
            x = layers.Concatenate()([x, dec_skip1])
            x = xception.sep_conv(x, 256,
                                  kernel_size=3,
                                  activation='relu')
            x = xception.sep_conv(x, 256,
                                  kernel_size=3,
                                  activation='relu')

        # you can use it with arbitary number of classes
        if self.classes == 21:
            last_layer_name = 'logits_semantic'
        else:
            last_layer_name = 'custom_logits_semantic'

        x = basic.ConvBlock(self.classes, kernel_size=1, padding='same',
                            name=last_layer_name).build(hp, inputs=x)
        x = layers.Lambda(lambda x: K.tf.image.resize_bilinear(
            x, size=(input_shape[0], input_shape[1])))(x)

        x = layers.Reshape((input_shape[0] * input_shape[1], self.classes))(x)
        x = layers.Activation('softmax')(x)
        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if inputs is not None:
            inputs = get_source_inputs(inputs)
        else:
            inputs = img_input

        model = Model(inputs, x, name='deeplabv3p')

        return model
