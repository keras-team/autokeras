class IOHyperModel(object):
    """A mixin class connecting the input nodes and heads with the adapters.

    This class is extended by the input nodes and the heads. The AutoModel calls the
    functions to get the corresponding adapters and pass the information back to the
    input nodes and heads.
    """

    def get_adapter(self):
        """Get the corresponding adapter.

        # Returns
            An instance of a subclass of autokeras.engine.Adapter.
        """
        raise NotImplementedError

    def config_from_adapter(self, adapter):
        """Load the learned information on dataset from the adapter.

        # Arguments
            adapter: An instance of a subclass of autokeras.engine.Adapter.
        """
        raise NotImplementedError
