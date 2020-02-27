class Serializable(object):
    """Serializable from and to JSON with same mechanism as Keras Layer."""

    def get_config(self):
        """Returns the current config of this object.

        # Returns
            Dictionary.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        """Build an instance from the config of this object.

        # Arguments
            config: Dict. The config of the object.
        """
        return cls(**config)
