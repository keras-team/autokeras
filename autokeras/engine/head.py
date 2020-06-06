import tensorflow as tf

from autokeras.engine import block as block_module
from autokeras.engine import io_hypermodel


def serialize_metrics(metrics):
    serialized = []
    for metric in metrics:
        if isinstance(metric, str):
            serialized.append([metric])
        else:
            serialized.append(tf.keras.metrics.serialize(metric))
    return serialized


def deserialize_metrics(metrics):
    deserialized = []
    for metric in metrics:
        if isinstance(metric, list):
            deserialized.append(metric[0])
        else:
            deserialized.append(tf.keras.metrics.deserialize(metric))
    return deserialized


def serialize_loss(loss):
    if isinstance(loss, str):
        return [loss]
    return tf.keras.losses.serialize(loss)


def deserialize_loss(loss):
    if isinstance(loss, list):
        return loss[0]
    return tf.keras.losses.deserialize(loss)


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
        self.loss = loss
        if metrics is None:
            metrics = []
        self.metrics = metrics
        # Mark if the head should directly output the input tensor.

    def get_config(self):
        config = super().get_config()
        config.update({
            'loss': serialize_loss(self.loss),
            'metrics': serialize_metrics(self.metrics),
            'output_shape': self.output_shape
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['loss'] = deserialize_loss(config['loss'])
        config['metrics'] = deserialize_metrics(config['metrics'])
        return super().from_config(config)

    def build(self, hp, inputs=None):
        raise NotImplementedError

    def config_from_adapter(self, adapter):
        self.output_shape = adapter.shape
