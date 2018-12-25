import os
import queue
import re
import time
import torch
import torch.multiprocessing as mp
import logging


from datetime import datetime
from autokeras.search import Searcher, train
from autokeras.constant import Constant
from autokeras.utils import verbose_print, get_system, assert_search_space


class Grid_Searcher(Searcher):
    def __init__(self, n_output_node, input_shape, path, metric, loss, generators, verbose, search_space={},
                 trainer_args=None):
        """Initialize the Searcher.

        Args:
            n_output_node: An integer, the number of classes.
            input_shape: A tuple. e.g. (28, 28, 1).
            path: A string. The path to the directory to save the searcher.
            metric: An instance of the Metric subclasses.
            loss: A function taking two parameters, the predictions and the ground truth.
            generators: A list of generators used to initialize the search.
            verbose: A boolean. Whether to output the intermediate information to stdout.
            search_space: Dictionary . Specifies the search dimensions and their possible values
            trainer_args: A dictionary. The params for the constructor of ModelTrainer.

         """
        if trainer_args is None:
            trainer_args = {}
        self.n_classes = n_output_node
        self.input_shape = input_shape
        self.verbose = verbose
        self.history = []
        self.path = path
        self.metric = metric
        self.loss = loss
        self.generators = generators
        self.model_count = 0
        self.descriptors = []
        self.trainer_args = trainer_args

        self.search_space, self.search_dimensions = assert_search_space(search_space)
        self.search_space_counter = 0
        if 'max_iter_num' not in self.trainer_args:
            self.trainer_args['max_iter_num'] = Constant.SEARCH_MAX_ITER
        self.training_queue = []
        logging.basicConfig(filename=self.path+datetime.now().strftime('run_%d_%m_%Y : _%H_%M.log'),
                            format='%(asctime)s - %(filename)s - %(message)s', level=logging.DEBUG)

    def init_search(self):
        """Call the generators to generate the initial architectures for the search."""
        if self.verbose:
            print('\nInitializing search.')
        for generator in self.generators:
            graph = generator(self.n_classes, self.input_shape). \
                generate(self.search_space[Constant.LENGTH_DIM][0], self.search_space[Constant.WIDTH_DIM][0])
            model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((graph, -1, model_id))
            self.descriptors.append(graph.extract_descriptor())

        if self.verbose:
            print('Initialization finished.')

    def search_space_exhausted(self):
        """ Check if Grid search has exhausted the search space """

        for i in range(len(self.search_space)):
            if self.search_dimensions[0][i] != self.search_dimensions[1][i]:
                return False
        return True

    def search(self, train_data, test_data, timeout=60 * 60 * 24):
        """Run the search loop of training, generating and updating once.

        The function will run the training and generate in parallel.
        Then it will update the controller.
        The training is just pop out a graph from the training_queue and train it.
        The generate will call teh self.generate function.
        The update will call the self.update function.

        Args:
            train_data: An instance of DataLoader.
            test_data: An instance of Dataloader.
            timeout: An integer, time limit in seconds.
        """

        start_time = time.time()
        torch.cuda.empty_cache()
        if not self.history:
            self.init_search()

        if self.search_space_exhausted():
            return
        # Start the new process for training.
        graph, other_info, model_id = self.training_queue.pop(0)
        if self.verbose:
            print('\n')
            print('+' + '-' * 46 + '+')
            print('|' + 'Training model {}'.format(model_id).center(46) + '|')
            print('+' + '-' * 46 + '+')
        # Temporary solution to support GOOGLE Colab
        if get_system() == Constant.SYS_GOOGLE_COLAB:
            ctx = mp.get_context('fork')
        else:
            ctx = mp.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=train, args=(q, graph, train_data, test_data, self.trainer_args,
                                            self.metric, self.loss, self.verbose, self.path))
        try:
            p.start()
            # Do the search in current thread.
            searched = False
            generated_graph = None
            generated_other_info = None
            if not self.training_queue:
                searched = True
                remaining_time = timeout - (time.time() - start_time)
                generated_other_info, generated_graph = self.generate(remaining_time, q)
                new_model_id = self.model_count
                self.model_count += 1
                self.training_queue.append((generated_graph, generated_other_info, new_model_id))
                self.descriptors.append(generated_graph.extract_descriptor())

            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                raise TimeoutError
            metric_value, loss, graph = q.get(timeout=remaining_time)

            if self.verbose and searched:
                verbose_print(generated_other_info, generated_graph, new_model_id)

            if metric_value is not None:
                self.add_model(metric_value, loss, graph, model_id)

        except (TimeoutError, queue.Empty) as e:
            raise TimeoutError from e
        finally:
            # terminate and join the subprocess to prevent any resource leak
            p.terminate()
            p.join()

    def generate(self, remaining_time, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            remaining_time: The remaining time in seconds.
            multiprocessing_queue: the Queue for multiprocessing return value.

        Returns:
            other_info: Always 0.
            generated_graph: An instance of Graph.

        """
        grid = self.get_grid()
        generated_graph = self.generators[0](self.n_classes, self.input_shape). \
            generate(grid[Constant.LENGTH_DIM], grid[Constant.WIDTH_DIM])
        return 0, generated_graph

    def get_grid(self):
        """ Return the next grid to be searched """

        self.search_space_counter += 1
        if self.search_space_counter < len(self.search_dimensions):
            return self.search_dimensions[self.search_space_counter]
        return None

