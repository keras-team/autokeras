##Searcher
Base class of all searcher class
This class is the base class of all searcher class, every searcher class can override its search function to implements its strategy
####Attributes
**n_classes**: number of classification

**input_shape**: Arbitrary, although all dimensions in the input shaped must be fixed.
           Use the keyword argument input_shape (tuple of integers, does not include the batch axis)
           when using this layer as the first layer in a model.

**verbose**: verbosity mode

**history_configs**: a list that stores all historical configuration

**history**: a list that stores the performance of model

**path**: place that store searcher

**model_count**: the id of model

###__init__
Init Searcher class with n_classes, input_shape, path, verbose
The Searcher will be loaded from file if it has been saved before.
###search
an search strategy that will be overridden by children classes

###load_best_model
return model with best accuracy

###add_model
add one model while will be trained to history list

##RandomSearcher
Random Searcher class inherited from ClassifierBase class
RandomSearcher implements its search function with random strategy
###__init__
Init RandomSearcher with n_classes, input_shape, path, verbose

###search
Override parent's search function. First model is randomly generated

##HillClimbingSearcher
HillClimbing Searcher class inherited from ClassifierBase class
HillClimbing Searcher implements its search function with hill climbing strategy
###__init__
Init HillClimbing Searcher with n_classes, input_shape, path, verbose

###_remove_duplicate
Remove the duplicate in the history_models

###search
Override parent's search function. First model is randomly generated

