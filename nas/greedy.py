import time
from copy import deepcopy

from autokeras.custom_queue import Queue
from autokeras.bayesian import contain, SearchTree
from autokeras.net_transformer import transform
from autokeras.search import Searcher

class GreedyOptimizer:

    def __init__(self, searcher, metric):
        self.searcher = searcher
        self.metric = metric

    def generate(self, descriptors, timeout, sync_message):
        """Generate new neighbor architectures from the best model.

        Args:
            descriptors: All the searched neural architectures.
            timeout: An integer. The time limit in seconds.
            sync_message: the Queue for multiprocessing return value.

        Returns:
            out: A list of 2-elements tuple. Each tuple contains
                an instance of Graph, a morphed neural network with weights
                and the father node id in the search tree.
        """
        out = []
        start_time = time.time()
        descriptors = deepcopy(descriptors)

        if isinstance(sync_message, Queue) and sync_message.qsize() != 0:
            return out
        model_id = self.searcher.get_neighbour_best_model_id()
        graph = self.searcher.load_model_by_id(model_id)
        father_id = model_id
        for temp_graph in transform(graph):
            if contain(descriptors, temp_graph.extract_descriptor()):
                continue
            out.append((deepcopy(temp_graph), father_id))
        remaining_time = timeout - (time.time() - start_time)

        if remaining_time < 0:
            raise TimeoutError
        return out


class GreedySearcher(Searcher):
    """ Class to search for neural architectures using Greedy search strategy.

    Attribute:
        optimizer: An instance of BayesianOptimizer.
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss, generators, verbose,
                 trainer_args=None,
                 default_model_len=None,
                 default_model_width=None):
        super(GreedySearcher, self).__init__(n_output_node, input_shape,
                                             path, metric, loss, generators,
                                             verbose, trainer_args, default_model_len,
                                             default_model_width)
        self.optimizer = GreedyOptimizer(self, metric)

    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.
                pass into the search algorithm for synchronizing

        Returns:
            results: A list of 2-element tuples. Each tuple contains an instance of Graph,
                and anything to be saved in the training queue together with the architecture

        """
        remaining_time = self._timeout - time.time()
        results = self.optimizer.generate(self.descriptors, remaining_time,
                                          multiprocessing_queue)
        if not results:
            new_father_id = 0
            generated_graph = self.generators[0](self.n_classes, self.input_shape). \
                generate(self.default_model_len, self.default_model_width)
            results.append((generated_graph, new_father_id))

        return results

    def update(self, other_info, model_id, graph, metric_value):
        return

    def load_neighbour_best_model(self):
        return self.load_model_by_id(self.get_neighbour_best_model_id())

    def get_neighbour_best_model_id(self):
        if self.metric.higher_better():
            return max(self.neighbour_history, key=lambda x: x['metric_value'])['model_id']
        return min(self.neighbour_history, key=lambda x: x['metric_value'])['model_id']
