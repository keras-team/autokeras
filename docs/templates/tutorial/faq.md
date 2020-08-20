## How to resume a previously killed run?
This feature is controlled by the `overwrite` argument of `AutoModel` or any other task APIs.
It is set to `False` by default,
which means it would not overwrite the contents of the directory.
In other words, it will continue the previous fit.

You can just run the same code again.
It will automatically resume the previously killed run.

## How to customize metrics and loss?
Please see the code example below.

```python
import autokeras as ak


clf = ak.ImageClassifier(
    max_trials=3,
    metrics=['mse'],
    loss='mse',
)
```

## How to use customized metrics to select the best model?
By default, AutoKeras use validation loss as the metric for selecting the best model.
Below is a code example of using customized metric for selecting models.
Please read the comments for the details.

```python
# Implement your customized metric according to the tutorial.
# https://keras.io/api/metrics/#creating-custom-metrics
import autokeras as ak


def f1_score(y_true, y_pred):
  ...

clf = ak.ImageClassifier(
    max_trials=3,

    # Wrap the function into a Keras Tuner Objective 
    # and pass it to AutoKeras.

    # Direction can be 'min' or 'max'
    # meaning we want to minimize or maximize the metric.

    # 'val_f1_score' is just add a 'val_' prefix
    # to the function name or the metric name.

    objective=kerastuner.Objective('val_f1_score', direction='min'),
    # Include it as one of the metrics.
    metrics=[f1_score],
)
```

## How to use multiple GPUs?
You can use the `distribution_strategy` argument when initializing any model you created with AutoKeras,
like AutoModel, ImageClassifier, StructuredDataRegressor and so on. This argument is supported by Keras Tuner.
AutoKeras supports the arguments supported by Keras Tuner.
Please see the discription of the argument [here](https://keras-team.github.io/keras-tuner/documentation/tuners/#tuner-class).

```python
import tensorflow as tf
import autokeras as ak


auto_model = ak.ImageClassifier(
    max_trials=3,
    distribution_strategy=tf.distribute.MirroredStrategy(),
)
```
