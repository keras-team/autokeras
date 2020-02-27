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
        self.loss = loss
        self.metrics = metrics
        # Mark if the head should directly output the input tensor.

    def get_config(self):
        config = super().get_config()
        config.update({
            'loss': self.loss,
            'metrics': self.metrics,
            'output_shape': self.output_shape
        })
        return config

    def build(self, hp, inputs=None):
        raise NotImplementedError

    def config_from_adapter(self, adapter):
        self.output_shape = adapter.shape
