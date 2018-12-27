import time
from copy import deepcopy
import multiprocessing as mp

from autokeras.bayesian import contain, SearchTree
from autokeras.net_transformer import transform


class GreedyOptimizer:

    def __init__(self, searcher, metric):
        self.searcher = searcher
        self.metric = metric
        self.search_tree = SearchTree()

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

        if isinstance(sync_message, mp.queues.Queue) and sync_message.qsize() != 0:
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

    def add_child(self, father_id, model_id):
        self.search_tree.add_child(father_id, model_id)