##BayesianSearcher
Base class of all searcher classes.
This class is the base class of all searcher classes, every searcher class can override its search function to implements its strategy.
#####Attributes
* **n_classes**: Number of classes in the traget classification task.

* **input_shape**: Arbitrary, although all dimensions in the input shaped must be fixed.
    Use the keyword argument `input_shape` (tuple of integers, does not include the batch axis)
    when using this layer as the first layer in a model.

* **verbose**: Verbosity mode.

* **history**: A list that stores the performance of model. Each element in it is a dictionary of 'model_id',
    'loss', and 'accuracy'.

* **path**: A string. The path to the directory for saving the searcher.

* **model_count**: An integer. the total number of neural networks in the current searcher.

* **descriptors**: A dictionary of all the neural network architectures searched.

* **trainer_args**: A dictionary. The params for the constructor of ModelTrainer.

* **default_model_len**: An integer. Number of convolutional layers in the initial architecture.

* **default_model_width**: An integer. The number of filters in each layer in the initial architecture.

* **gpr**: A GaussianProcessRegressor for bayesian optimization.

* **search_tree**: The data structure for storing all the searched architectures in tree structure.

* **training_queue**: A list of the generated architectures to be trained.

* **x_queue**: A list of trained architectures not updated to the gpr.

* **y_queue**: A list of trained architecture performances not updated to the gpr.

* **beta**: A float. The beta in the UCB acquisition function.

* **t_min**: A float. The minimum temperature during simulated annealing.

###__init__
Initialize the BayesianSearcher.

#####Args
* **n_classes**: An integer, the number of classes.

* **input_shape**: A tuple. e.g. (28, 28, 1).

* **path**: A string. The path to the directory to save the searcher.

* **verbose**: A boolean. Whether to output the intermediate information to stdout.

* **trainer_args**: A dictionary. The params for the constructor of ModelTrainer.

* **default_model_len**: An integer. Number of convolutional layers in the initial architecture.

* **default_model_width**: An integer. The number of filters in each layer in the initial architecture.

* **beta**: A float. The beta in the UCB acquisition function.

* **kernel_lambda**: A float. The balance factor in the neural network kernel.

* **t_min**: A float. The minimum temperature during simulated annealing.

