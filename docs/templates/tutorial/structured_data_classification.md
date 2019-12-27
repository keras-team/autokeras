# Structured Data Classification
## A Simple Example
The first step is to prepare your data. Here we use the [Titanic
dataset](https://www.kaggle.com/c/titanic) as an example. You can download the CSV
files [here](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification).

The second step is to run the
[StructuredDataClassifier](/structured_data_classifier).
Replace all the `/path/to` with the path to the csv files.

```python
import autokeras as ak

# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(max_trials=10) # It tries 10 different models.
# Feed the structured data classifier with training data.
clf.fit(
    # The path to the train.csv file.
    '/path/to/train.csv',
    # The name of the label column.
    'survived')
# Predict with the best model.
predicted_y = clf.predict('/path/to/eval.csv')
# Evaluate the best model with testing data.
print(clf.evaluate('/path/to/eval.csv', 'survived'))
```

## Data Format
The AutoKeras StructuredDataClassifier is quite flexible for the data format.

The example above shows how to use the CSV files directly. Besides CSV files, it also
supports numpy.ndarray, pandas.DataFrame or tensorflow.Dataset. The data should be
two-dimensional with numerical or categorical values.  For the classification labels,
AutoKeras accepts both plain labels, i.e.  strings or integers, and one-hot encoded
encoded labels, i.e. vectors of 0s and 1s.  Since IMDB dataset is binary
classification, it should not be one-hot encoded.

We also support using [tf.data.Dataset](
https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=stable) format for
the training data. The labels have to be one-hot encoded.  So you can wrap the data
above into tensorflow Dataset as follows.

```python
import tensorflow as tf
train_set = tf.data.Dataset.from_tensor_slices(((x_train, ), (y_train, )))
test_set = tf.data.Dataset.from_tensor_slices(((x_test, ), (y_test, )))

clf = ak.TextClassifier(max_trials=10)
# Feed the tensorflow Dataset to the classifier.
clf.fit(train_set)
# Predict with the best model.
predicted_y = clf.predict(test_set)
# Evaluate the best model with testing data.
print(clf.evaluate(test_set))
```

## Validation Data
By default, AutoKeras use the last 20% of training data as validation data.
As shown in the example below, you can use `validation_split` to specify the percentage.

```python
clf.fit(x_train,
        y_train,
        # Split the training data and use the last 15% as validation data.
        validation_split=0.15)
```

You can also use your own validation set
instead of splitting it from the training data with `validation_data`.

```python
split = 5000
x_val = x_train[split:]
y_val = y_train[split:]
x_train = x_train[:split]
y_train = y_train[:split]
clf.fit(x_train,
        y_train,
        # Use your own validation set.
        validation_data=(x_val, y_val))
```

## Customized Search Space
For advanced users, you may customize your search space by using
[AutoModel](/auto_model/#automodel-class) instead of
[TextClassifier](/text_classifier). You can configure the
[TextBlock](/block/#textblock-class) for some high-level configurations, `vectorizer`
for the type of text vectorization method to use.  You can use 'sequence', which uses
[TextToInteSequence](/preprocessor/#texttointsequence-class) to convert the words to
integers and use [EmbeddingBlock](/block/#embeddingblock-class) for embedding the
integer sequences, or you can use 'ngram', which uses
[TextToNgramVector](/preprocessor/#texttongramvector-class) to vectorize the
sentences.  You can also do not specify these arguments, which would leave the
different choices to be tuned automatically.  See the following example for detail.

```python
import autokeras as ak

input_node = ak.TextInput()
output_node = ak.TextBlock(vectorizer='ngram')(input_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
clf.fit(x_train, y_train)
```
The usage of [AutoModel](/auto_model/#automodel-class) is similar to the
[functional API](https://www.tensorflow.org/guide/keras/functional) of Keras.
Basically, you are building a graph, whose edges are blocks and the nodes are intermediate outputs of blocks.
To add an edge from `input_node` to `output_node` with
`output_node = ak.[some_block]([block_args])(input_node)`.

You can even also use more fine grained blocks to customize the search space even
further. See the following example.

```python
import autokeras as ak

input_node = ak.TextInput()
output_node = preprocessor_module.TextToIntSequence()(input_node)
output_node = block_module.EmbeddingBlock()(output_node)
# Use separable Conv layers in Keras.
output_node = block_module.ConvBlock(separable=True)(output_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
clf.fit(x_train, y_train)
```


## Reference
[TextClassifier](/text_classifier),
[AutoModel](/auto_model/#automodel-class),
[TextBlock](/block/#textblock-class),
[TextToInteSequence](/preprocessor/#texttointsequence-class)
[EmbeddingBlock](/block/#embeddingblock-class) 
[TextToNgramVector](/preprocessor/#texttongramvector-class) 
[ConvBlock](/block/#convblock-class)
[ClassificationHead](/head/#classification-head-class)
