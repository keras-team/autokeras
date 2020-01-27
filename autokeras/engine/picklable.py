import pickle

import tensorflow as tf
from kerastuner.engine import stateful


class Picklable(stateful.Stateful):
    """The mixin for saving and loading config and weights for HyperModels.

    We define weights for any hypermodel as something that can only be know after
    seeing the data. The rest of the states are configs.
    """

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

    def save(self, fname):
        """Save state to file.

        # Arguments
            fname: String. The path to a file to save the state.
        """
        state = self.get_state()
        with tf.io.gfile.GFile(fname, 'wb') as f:
            pickle.dump(state, f)
        return str(fname)

    def reload(self, fname):
        """Load state from file.

        # Arguments
            fname: String. The path to a file to load the state.
        """
        with tf.io.gfile.GFile(fname, 'rb') as f:
            state = pickle.load(f)
        self.set_state(state)
