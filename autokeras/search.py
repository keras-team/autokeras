import os
import queue
import re
import time
import torch
import torch.multiprocessing as mp

from autokeras.bayesian import BayesianOptimizer
from autokeras.constant import Constant
from autokeras.nn.model_trainer import ModelTrainer
from autokeras.utils import pickle_to_file, pickle_from_file, verbose_print, get_system


class Searcher:
    """Class to search for neural architectures.

    This class generate new architectures, call the trainer to train it, and update the Bayesian optimizer.

    Attributes:
        n_classes: Number of classes in the target classification task.
        input_shape: Arbitrary, although all dimensions in the input shaped must be fixed.
            Use the keyword argument `input_shape` (tuple of integers, does not include the batch axis)
            when using this layer as the first layer in a model.
        verbose: Verbosity mode.
        history: A list that stores the performance of model. Each element in it is a dictionary of 'model_id',
            'loss', and 'metric_value'.
        path: A string. The path to the directory for saving the searcher.
        metric: An instance of the Metric subclasses.
        loss: A function taking two parameters, the predictions and the ground truth.
        generators: A list of generators used to initialize the search.
        model_count: An integer. the total number of neural networks in the current searcher.
        descriptors: A dictionary of all the neural network architectures searched.
        trainer_args: A dictionary. The params for the constructor of ModelTrainer.
        default_model_len: An integer. Number of convolutional layers in the initial architecture.
        default_model_width: An integer. The number of filters in each layer in the initial architecture.
        training_queue: A list of the generated architectures to be trained.
        x_queue: A list of trained architectures not updated to the gpr.
        y_queue: A list of trained architecture performances not updated to the gpr.
        beta: A float. The beta in the UCB acquisition function.
        t_min: A float. The minimum temperature during simulated annealing.
        bo: An instance of BayesianOptimizer.
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss, generators, verbose,
                 trainer_args=None,
                 default_model_len=Constant.MODEL_LEN,
                 default_model_width=Constant.MODEL_WIDTH,
                 beta=Constant.BETA,
                 t_min=None):
        """Initialize the Searcher.

        Args:
            n_output_node: An integer, the number of classes.
            input_shape: A tuple. e.g. (28, 28, 1).
            path: A string. The path to the directory to save the searcher.
            metric: An instance of the Metric subclasses.
            loss: A function taking two parameters, the predictions and the ground truth.
            generators: A list of generators used to initialize the search.
            verbose: A boolean. Whether to output the intermediate information to stdout.
            trainer_args: A dictionary. The params for the constructor of ModelTrainer.
            default_model_len: An integer. Number of convolutional layers in the initial architecture.
            default_model_width: An integer. The number of filters in each layer in the initial architecture.
            beta: A float. The beta in the UCB acquisition function.
            t_min: A float. The minimum temperature during simulated annealing.
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
        self.default_model_len = default_model_len
        self.default_model_width = default_model_width
        if 'max_iter_num' not in self.trainer_args:
            self.trainer_args['max_iter_num'] = Constant.SEARCH_MAX_ITER

        self.training_queue = []
        self.x_queue = []
        self.y_queue = []
        if t_min is None:
            t_min = Constant.T_MIN
        self.bo = BayesianOptimizer(self, t_min, metric, beta)

    def load_model_by_id(self, model_id):
        return pickle_from_file(os.path.join(self.path, str(model_id) + '.graph'))

    def load_best_model(self):
        return self.load_model_by_id(self.get_best_model_id())

    def get_metric_value_by_id(self, model_id):
        for item in self.history:
            if item['model_id'] == model_id:
                return item['metric_value']
        return None

    def get_best_model_id(self):
        if self.metric.higher_better():
            return max(self.history, key=lambda x: x['metric_value'])['model_id']
        return min(self.history, key=lambda x: x['metric_value'])['model_id']

    def replace_model(self, graph, model_id):
        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.graph'))

    def add_model(self, metric_value, loss, graph, model_id):
        """Append the information of evaluated architecture to history."""
        if self.verbose:
            print('\nSaving model.')

        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.graph'))

        ret = {'model_id': model_id, 'loss': loss, 'metric_value': metric_value}
        self.history.append(ret)

        # Update best_model text file
        if model_id == self.get_best_model_id():
            file = open(os.path.join(self.path, 'best_model.txt'), 'w')
            file.write('best model: ' + str(model_id))
            file.close()

        if self.verbose:
            idx = ['model_id', 'loss', 'metric_value']
            header = ['Model ID', 'Loss', 'Metric Value']
            line = '|'.join(x.center(24) for x in header)
            print('+' + '-' * len(line) + '+')
            print('|' + line + '|')

            if self.history:
                r = self.history[-1]
                print('+' + '-' * len(line) + '+')
                line = '|'.join(str(r[x]).center(24) for x in idx)
                print('|' + line + '|')
            print('+' + '-' * len(line) + '+')

        return ret

    def init_search(self):
        """Call the generators to generate the initial architectures for the search."""
        if self.verbose:
            print('\nInitializing search.')
        for generator in self.generators:
            graph = generator(self.n_classes, self.input_shape). \
                generate(self.default_model_len, self.default_model_width)
            model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((graph, -1, model_id))
            self.descriptors.append(graph.extract_descriptor())
        # if graph is not None and model_id is not None:
        #     for child_graph in default_transform(graph):
        #         child_id = self.model_count
        #         self.model_count += 1
        #         self.training_queue.append((child_graph, model_id, child_id))
        #         self.descriptors.append(child_graph.extract_descriptor())

        if self.verbose:
            print('Initialization finished.')

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
                generated_other_info, generated_graph = self.generate(remaining_time)
                new_model_id = self.model_count
                self.model_count += 1
                self.training_queue.append((generated_graph, generated_other_info, new_model_id))
                self.descriptors.append(generated_graph.extract_descriptor())

            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                raise TimeoutError
            metric_value, loss, graph = q.get(timeout=remaining_time)

            if self.verbose and searched:
                verbose_print(generated_other_info, generated_graph)

            self.add_model(metric_value, loss, graph, model_id)
            self.update(other_info, graph, metric_value, model_id)

            self.export_json(os.path.join(self.path, 'history.json'))

        except (TimeoutError, queue.Empty) as e:
            raise TimeoutError from e
        except RuntimeError as e:
            if not re.search('out of memory', str(e)):
                raise e
            if self.verbose:
                print('\nCurrent model size is too big. Discontinuing training this model to search for other models.')
            Constant.MAX_MODEL_SIZE = graph.size() - 1
            return
        finally:
            # terminate and join the subprocess to prevent any resource leak
            p.terminate()
            p.join()

    def update(self, other_info, graph, metric_value, model_id):
        """ Update the controller with evaluation result of a neural architecture.

        Args:
            other_info: Anything. In our case it is the father ID in the search tree.
            graph: An instance of Graph. The trained neural architecture.
            metric_value: The final evaluated metric value.
            model_id: An integer.
        """
        father_id = other_info
        self.bo.fit([graph.extract_descriptor()], [metric_value])
        self.bo.add_child(father_id, model_id)

    def generate(self, remaining_time):
        """Generate the next neural architecture.

        Args:
            remaining_time: The remaining time in seconds.

        Returns:
            other_info: Anything to be saved in the training queue together with the architecture.
            generated_graph: An instance of Graph.

        """
        generated_graph, new_father_id = self.bo.generate(self.descriptors,
                                                          remaining_time)
        if new_father_id is None:
            new_father_id = 0
            generated_graph = self.generators[0](self.n_classes, self.input_shape). \
                generate(self.default_model_len, self.default_model_width)

        return new_father_id, generated_graph

    def export_json(self, path):
        """Export a json file of the search process."""
        data = dict()

        networks = []
        for model_id in range(self.model_count - len(self.training_queue)):
            networks.append(self.load_model_by_id(model_id).extract_descriptor().to_json())

        tree = self.bo.search_tree.get_dict()

        # Saving the data to file.
        data['networks'] = networks
        data['tree'] = tree
        import json
        with open(path, 'w') as fp:
            json.dump(data, fp)


def train(q, graph, train_data, test_data, trainer_args, metric, loss, verbose, path):
    """Train the neural architecture."""
    model = graph.produce_model()
    loss, metric_value = ModelTrainer(model=model,
                                      path=path,
                                      train_data=train_data,
                                      test_data=test_data,
                                      metric=metric,
                                      loss_function=loss,
                                      verbose=verbose).train_model(**trainer_args)
    model.set_weight_to_graph()
    if q:
        q.put((metric_value, loss, model.graph))
    return metric_value, loss, model.graph
