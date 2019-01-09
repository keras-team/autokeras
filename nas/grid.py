from autokeras.constant import Constant
from autokeras.search import Searcher
from autokeras.utils import assert_search_space


class GridSearcher(Searcher):
    """ Class to search for neural architectures using Greedy search strategy.

    Attribute:
        search_space: A dictionary. Specifies the search dimensions and their possible values
    """
    def __init__(self, n_output_node, input_shape, path, metric, loss, generators, verbose, search_space={},
                 trainer_args=None, default_model_len=None, default_model_width=None):
        super(GridSearcher, self).__init__(n_output_node, input_shape, path, metric, loss, generators, verbose,
                                           trainer_args, default_model_len, default_model_width)
        self.search_space, self.search_dimensions = assert_search_space(search_space)
        self.search_space_counter = 0

    def get_search_dimensions(self):
        return self.search_dimensions

    def search_space_exhausted(self):
        """ Check if Grid search has exhausted the search space """
        if self.search_space_counter == len(self.search_dimensions):
            return True
        return False

    def search(self, train_data, test_data, timeout=60 * 60 * 24):
        """Run the search loop of training, generating and updating once.

        Call the base class implementation for search with

        Args:
            train_data: An instance of DataLoader.
            test_data: An instance of Dataloader.
            timeout: An integer, time limit in seconds.
        """
        if self.search_space_exhausted():
            return
        else:
            super().search(train_data, test_data, timeout)

    def update(self, other_info, model_id, graph, metric_value):
        return

    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.

        Returns:
            list of 2-element tuples: generated_graph and other_info,
            for grid searcher the length of list is 1.
            generated_graph: An instance of Graph.
            other_info: Always 0.
        """
        grid = self.get_grid()
        self.search_space_counter += 1
        generated_graph = self.generators[0](self.n_classes, self.input_shape). \
            generate(grid[Constant.LENGTH_DIM], grid[Constant.WIDTH_DIM])
        return [(generated_graph, 0)]

    def get_grid(self):
        """ Return the next grid to be searched """
        if self.search_space_counter < len(self.search_dimensions):
            return self.search_dimensions[self.search_space_counter]
        return None