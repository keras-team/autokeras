import tensorflow as tf

from autokeras.engine import block as block_module
from autokeras.engine import io_hypermodel


class Head(block_module.Block, io_hypermodel.IOHyperModel):
    """Base class for the heads, e.g. classification, regression.

    # Arguments
        loss: A Keras loss function. Defaults to None. If None, the loss will be
            inferred from the AutoModel.
        metrics: A list of Keras metrics. Defaults to None. If None, the metrics will
            be inferred from the AutoModel.
        output_shape: Tuple of int(s). Defaults to None. If None, the output shape
            will be inferred from the AutoModel.
    """

    def __init__(self, loss=None, metrics=None, output_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape
        self.loss = tf.keras.losses.get(loss)
        if metrics is None:
            metrics = []
        self.metrics = [tf.keras.metrics.get(metric) for metric in metrics]
        # Mark if the head should directly output the input tensor.

    def get_config(self):
        config = super().get_config()
        config.update({
            'loss': tf.keras.losses.serialize(self.loss),
            'metrics': [tf.keras.metrics.serialize(metric)
                        for metric in self.metrics],
            'output_shape': self.output_shape
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['loss'] = tf.keras.losses.deserialize(config['loss'])
        config['metrics'] = [tf.keras.metrics.deserialize(metric)
                             for metric in config['metrics']]
        return super().from_config(config)

    def build(self, hp, inputs=None):
        raise NotImplementedError

    def config_from_adapter(self, adapter):
        self.output_shape = adapter.shape
