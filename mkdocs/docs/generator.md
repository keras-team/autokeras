##ClassifierGenerator
The base class of classifier generators.
ClassifierGenerator is the base class of all classifier generator classes. It is used for generating classifier models.
####Attributes
**n_classes**: Number of classes in the input data.

**input_shape**: A tuple of integers containing the size of each dimension of the input data,
    excluding the dimension of number of training examples. The length of the tuple should
    between two and four inclusively.

###__init__
###_get_pool_layer_func
Return MaxPooling function based on the dimension of input shape.

###_get_shape
Return filter shape tuple based on the dimension of input shape.

##DefaultClassifierGenerator
A classifier generator always generates models with the same default architecture and configuration.

###__init__
###generate
Return the default classifier model that has been compiled.

##RandomConvClassifierGenerator
A classifier generator that generates random convolutional neural networks.

###__init__
###generate
Return the random generated CNN model.

