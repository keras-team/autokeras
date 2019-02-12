from random import randrange

from autokeras.bayesian import SearchTree, contain
from autokeras.net_transformer import transform
from autokeras.search import Searcher


class RandomSearcher(Searcher):
    """ Class to search for neural architectures using Random search strategy.
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss, generators, verbose,
                 trainer_args=None,
                 default_model_len=None,
                 default_model_width=None):
        super(RandomSearcher, self).__init__(n_output_node, input_shape,
                                             path, metric, loss, generators,
                                             verbose, trainer_args, default_model_len,
                                             default_model_width)

    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.

        Returns:
            list of 2-element tuples: generated_graph and other_info,
            for random searcher the length of list is 1.
            generated_graph: An instance of Graph.
            other_info: Anything to be saved in the training queue together with the architecture.

        """
        random_index = randrange(len(self.history))
        model_id = self.history[random_index]['model_id']
        graph = self.load_model_by_id(model_id)
        new_father_id = None
        generated_graph = None
        for temp_graph in transform(graph):
            if not contain(self.descriptors, temp_graph.extract_descriptor()):
                new_father_id = model_id
                generated_graph = temp_graph
                break
        if new_father_id is None:
            new_father_id = 0
            generated_graph = self.generators[0](self.n_classes, self.input_shape). \
                generate(self.default_model_len, self.default_model_width)

        return [(generated_graph, new_father_id)]

    def update(self, other_info, model_id, graph, metric_value):
        return