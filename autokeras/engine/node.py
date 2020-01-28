from autokeras.engine import picklable


class Node(picklable.Picklable):
    """The nodes in a network connecting the blocks."""

    def __init__(self, shape=None):
        super().__init__()
        self.in_blocks = []
        self.out_blocks = []
        self.shape = shape

    def add_in_block(self, hypermodel):
        self.in_blocks.append(hypermodel)

    def add_out_block(self, hypermodel):
        self.out_blocks.append(hypermodel)

    def build(self):
        raise NotImplementedError

    def get_config(self):
        return {}

    def get_state(self):
        return {'shape': self.shape}

    def set_state(self, state):
        self.shape = state['shape']

