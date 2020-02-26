from autokeras.tuners import greedy

IMAGE_CLASSIFIER = [{
    'image_block_1/block_type': 'vanilla',
    'image_block_1/normalize': True,
    'image_block_1/augment': False,
    'image_block_1/vanilla_1/kernel_size': 3,
    'image_block_1/vanilla_1/num_blocks': 1,
    'image_block_1/vanilla_1/separable': False,
    'image_block_1/vanilla_1/dropout_rate': 0.25,
    'image_block_1/vanilla_1/filters_0_1': 32,
    'image_block_1/vanilla_1/filters_0_2': 64,
    'spatial_reduction_1/reduction_type': 'flatten',
    'dense_block_1/num_layers': 1,
    'dense_block_1/use_batchnorm': False,
    'dense_block_1/dropout_rate': 0,
    'dense_block_1/units_0': 128,
    'classification_head_1/dropout_rate': 0.5,
    'optimizer': 'adam'
}, {
    'image_block_1/block_type': 'resnet',
    'image_block_1/normalize': True,
    'image_block_1/augment': True,
    'image_block_1/resnet_1/version': 'v2',
    'image_block_1/resnet_1/pooling': 'avg',
    'image_block_1/resnet_1/conv3_depth': 4,
    'image_block_1/resnet_1/conv4_depth': 6,
    'dense_block_1/num_layers': 2,
    'dense_block_1/use_batchnorm': False,
    'dense_block_1/dropout_rate': 0,
    'dense_block_1/units_0': 32,
    'dense_block_1/units_1': 32,
    'classification_head_1/dropout_rate': 0,
    'optimizer': 'adam'
}]
IMAGE_SEGMENTER = [{

}]

class ImageClassifierTuner(greedy.Greedy):
    def __init__(self, **kwargs):
        super().__init__(
            initial_hps=IMAGE_CLASSIFIER,
            **kwargs)

class ImageSetmenterTuner(greedy.Greedy):
    def __init__(self, **kwargs):
        super().__init__(
            initial_hps=IMAGE_SEGMENTER,
            **kwargs)