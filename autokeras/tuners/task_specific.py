from autokeras.tuners import greedy

IMAGE_CLASSIFIER = [{
    'image_block_1/block_type': 'vanilla',
    'image_block_1/normalize': True,
    'image_block_1/augment': False,
    'image_block_1/conv_block_1/kernel_size': 3,
    'image_block_1/conv_block_1/num_blocks': 1,
    'image_block_1/conv_block_1/num_layers': 2,
    'image_block_1/conv_block_1/max_pooling': True,
    'image_block_1/conv_block_1/separable': False,
    'image_block_1/conv_block_1/dropout_rate': 0.25,
    'image_block_1/conv_block_1/filters_0_0': 32,
    'image_block_1/conv_block_1/filters_0_1': 64,
    'classification_head_1/spatial_reduction_1/reduction_type': 'flatten',
    'classification_head_1/dropout_rate': 0.5,
    'optimizer': 'adam'
}, {
    'image_block_1/block_type': 'resnet',
    'image_block_1/normalize': True,
    'image_block_1/augment': True,
    'image_block_1/image_augmentation_1/horizontal_flip': True,
    'image_block_1/image_augmentation_1/vertical_flip': True,
    'image_block_1/res_net_block_1/version': 'v2',
    'image_block_1/res_net_block_1/pooling': 'avg',
    'image_block_1/res_net_block_1/conv3_depth': 4,
    'image_block_1/res_net_block_1/conv4_depth': 6,
    'classification_head_1/dropout_rate': 0,
    'optimizer': 'adam'
}]

TEXT_CLASSIFIER = [{
    'text_block_1/vectorizer': 'sequence',
    'classification_head_1/dropout_rate': 0,
    'optimizer': 'adam',
    'text_block_1/max_tokens': 5000,
    'text_block_1/conv_block_1/separable': False,
    'text_block_1/text_to_int_sequence_1/output_sequence_length': 512,
    'text_block_1/embedding_1/pretraining': 'none',
    'text_block_1/embedding_1/embedding_dim': 64,
    'text_block_1/embedding_1/dropout_rate': 0.25,
    'text_block_1/conv_block_1/kernel_size': 5,
    'text_block_1/conv_block_1/num_blocks': 1,
    'text_block_1/conv_block_1/num_layers': 1,
    'text_block_1/conv_block_1/max_pooling': False,
    'text_block_1/conv_block_1/dropout_rate': 0,
    'text_block_1/conv_block_1/filters_0_0': 256,
    'text_block_1/spatial_reduction_1/reduction_type': 'global_max',
    'text_block_1/dense_block_1/num_layers': 1,
    'text_block_1/dense_block_1/use_batchnorm': False,
    'text_block_1/dense_block_1/dropout_rate': 0.5,
    'text_block_1/dense_block_1/units_0': 256,
}]


class ImageClassifierTuner(greedy.Greedy):
    def __init__(self, **kwargs):
        super().__init__(
            initial_hps=IMAGE_CLASSIFIER,
            **kwargs)


class TextClassifierTuner(greedy.Greedy):
    def __init__(self, **kwargs):
        super().__init__(
            initial_hps=TEXT_CLASSIFIER,
            **kwargs)
