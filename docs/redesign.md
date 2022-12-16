# AutoKeras Redesign

## Motivation
redesign how the graph can tune preprocessor, model, and post-processor
altogether. This is mainly to adapt to the latest best practices of Keras,
which do not include the preprocessing layers in the main model.

## The problem

The following problems should all be resolved by the new design.
Some of the problems are supported by AutoKeras today, but some are not.

* Hyperparameters exists in both preprocessors.
    * Example: Data augmentation contains hyperparameters.
* The preprocessors needs to know if it is training, validation, or testing.
    * Example: Data augmentation only apply when training.
* The hyperparameter for model selection is deciding the preprocessors.
    * Example: If choose BERT, the preprocessor should use `BertTokenizer`.
* The preprocessor need to access the target (y) data.
    * Example: Feature selection based on y.
    * Example: Object detection preprocessing the images.
* All the preprocessors applied needs to be tracked to export the preprocessors.
* Post-processor needs information from preprocessors.
    * Example: Post-processor to decode probability vector back to
      classification label needs information from the y encoder.

## The current design

It stacks the preprocessors based on input type.
Do not select preprocessors based on the hyperparameter value for model selection.


## Infeasible best solution

The ideal solution would be separate the search space representation (the
`HyperModel`s) from the actual implementation (the `tf.data` operations and
`tf.keras.Model`). We just use the `HyperModel` to get all the hyperparameter
values and construct an abstraction of the actual implementation, which may also
be called an intermediate representation (IR), and build the IR into the actual
implementation. It sounds it would be easy to separate the `tf.data` operations
and the `tf.keras.Model`. However, this is not possible to implement without
adding significant burden for all the `autokeras.Block` subclasses
implementations. It is because KerasTuner create hyperparameters as it builds
the `HyperModel`. So, it is not possible to get all the hyperparameter values
without running the code of actual `tf.data` operations and `tf.keras.Model`
building.

## The new design

### Mixing the preprocessors and model

No longer create the preprocessors based on the input nodes.
Make preprocessors part of the graph.
Each block can be build into a mix of preprocessors and parts of the model.  The
input to `Block.build()` can be either a `KerasTensor` or `tf.data.Dataset`.
Whenever we need to switch a `Dataset` into a `KerasTensor`, we register the
`keras.Input` in the graph, so that we can use it to build the `keras.Model`
later. Similar to the Keras saving mechanism of custom objects registered in
[`_GLOBAL_CUSTOM_OBJECTS`](https://github.com/keras-team/keras/blob/v2.11.0/keras/saving/object_registration.py#L23).

How to register?
Using `Block._build_wrapper()`. If a dataset is passed to a block that suppose
to be part of the Keras model, it should do the switch from dataset to Keras
input.
In `Block._build_wrapper()`, it should try to modify some constant value from
another module to put the input node in.

Note: we require, wherever switching from dataset to a Keras input, it has to
rely on the `Block._build_wrapper()` to do the job.

### Export the preprocessor

We keep the current design of building `HyperPreprocessor` into `Preprocessor`.
Whenever a preprocessing is done on the `Dataset`, we record the input and
output dataset and the preprocessor, so that we can track and reconstruct the
preprocessing computation graph (not rebuilding the preprocessors from the
`HyperProcessor`s because fit preprocessors takes time. We can directly use the
already built preprocessors) and export the prerpocessors.  The current saving
mechanism of the preprocessors can also be kept.

How to track?
Similar to register `input`s above. In `HyperPreprocessor._build_wrapper()`, it
register the input and output datasets and the preprocessor built processed
them.

### About handling the target values

`y` can be either treat separately or also be part of the graph. Treating
separately would be much easier. The preprocessing and postprocessing needs to
share info, but they don't share info with the rest of the graph. Some
preprocessors needs `y`.

The analyzer and heads information flow can be kept without touch. The analyzer
analyzes the data and pass the configs into the heads, for example, use sigmoid
or softmax for classification.

However, the coupling between heads and preprocessing and postprocessing of `y`
may be decoupled. The information needed for pre & post processing came from the
analysers. No need to route it into the heads. Routing through heads would make
implementing custom heads harder.

Although, we get the analysers by the heads, no information is actually passed
from the heads to the analysers. This design is only because the users are
specifying the `y`s using the heads.

### Reduce unnecessary dataset conversions

Have more strict type and shape checks, which reduce the overhead of the
preprocessors to reshape and type convert the dataset.  First, analyze the
dataset. If doesn't meet the requirements, raise a clear error.

### Accomendations in KerasTuner

Seems no accomendations needed. Should continue to override
`Tuner._build_and_fit_model()`.

## TODOs

* Remove prototype directory.
* Remove prototype from coverage exclusion in setup.cfg.