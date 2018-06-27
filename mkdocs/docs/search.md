##BayesianSearcher
Base class of all searcher class
This class is the base class of all searcher class, every searcher class can override its search function to implements its strategy
####Attributes
**n_classes**: number of classification

**input_shape**: Arbitrary, although all dimensions in the input shaped must be fixed.
           Use the keyword argument input_shape (tuple of integers, does not include the batch axis)
           when using this layer as the first layer in a model.

**verbose**: verbosity mode

**history**: a list that stores the performance of model

**path**: place that store searcher

**model_count**: the id of model

###__init__
Init Searcher class with n_classes, input_shape, path, verbose
The Searcher will be loaded from file if it has been saved before.
###load_model_by_id
###load_best_model
return model with best accuracy

###get_accuracy_by_id
###get_best_model_id
###replace_model
###add_model
add one model while will be trained to history list

####Returns
History object.
###init_search
###search
###maximize_acq
###acq
##SearchTree
###__init__
###add_child
###get_leaves
##Elem
###__init__
###__eq__
###__lt__
