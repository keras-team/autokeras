"""shell
pip install autokeras
"""

"""
## A Simple Example
The first step is to prepare your data. Here we use the [IMDB
dataset](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification) as
an example.
"""

import numpy as np
from tensorflow.keras.datasets import imdb

# Load the integer sequence the IMDB dataset with Keras.
index_offset = 3  # word index offset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000,
                                                      index_from=index_offset)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# Prepare the dictionary of index to word.
word_to_id = imdb.get_word_index()
word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
id_to_word = {value: key for key, value in word_to_id.items()}
# Convert the word indices to words.
x_train = list(map(lambda sentence: ' '.join(
    id_to_word[i] for i in sentence), x_train))
x_test = list(map(lambda sentence: ' '.join(
    id_to_word[i] for i in sentence), x_test))
x_train = np.array(x_train, dtype=np.str)
x_test = np.array(x_test, dtype=np.str)
print(x_train.shape)  # (25000,)
print(y_train.shape)  # (25000, 1)
print(x_train[0][:50])  # <START> this film was just brilliant casting <UNK>

"""
The second step is to run the [TextClassifier](/text_classifier).
"""

import autokeras as ak

# Initialize the text classifier.
clf = ak.TextClassifier(max_trials=1) # It tries 10 different models.
# Feed the text classifier with training data.
clf.fit(x_train, y_train, epochs=2)
# Predict with the best model.
predicted_y = clf.predict(x_test)
# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))


"""
## Validation Data
By default, AutoKeras use the last 20% of training data as validation data.
As shown in the example below, you can use `validation_split` to specify the percentage.
"""

clf.fit(x_train,
        y_train,
        # Split the training data and use the last 15% as validation data.
        validation_split=0.15)

"""
You can also use your own validation set
instead of splitting it from the training data with `validation_data`.
"""

split = 5000
x_val = x_train[split:]
y_val = y_train[split:]
x_train = x_train[:split]
y_train = y_train[:split]
clf.fit(x_train,
        y_train,
        epochs=2,
        # Use your own validation set.
        validation_data=(x_val, y_val))

"""
## Customized Search Space
For advanced users, you may customize your search space by using
[AutoModel](/auto_model/#automodel-class) instead of
[TextClassifier](/text_classifier). You can configure the
[TextBlock](/block/#textblock-class) for some high-level configurations, e.g., `vectorizer`
for the type of text vectorization method to use.  You can use 'sequence', which uses
[TextToInteSequence](/preprocessor/#texttointsequence-class) to convert the words to
integers and use [Embedding](/block/#embedding-class) for embedding the
integer sequences, or you can use 'ngram', which uses
[TextToNgramVector](/preprocessor/#texttongramvector-class) to vectorize the
sentences.  You can also do not specify these arguments, which would leave the
different choices to be tuned automatically.  See the following example for detail.
"""

import autokeras as ak

input_node = ak.TextInput()
output_node = ak.TextBlock(vectorizer='ngram')(input_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=1)
clf.fit(x_train, y_train, epochs=2)

"""
The usage of [AutoModel](/auto_model/#automodel-class) is similar to the
[functional API](https://www.tensorflow.org/guide/keras/functional) of Keras.
Basically, you are building a graph, whose edges are blocks and the nodes are intermediate outputs of blocks.
To add an edge from `input_node` to `output_node` with
`output_node = ak.[some_block]([block_args])(input_node)`.

You can even also use more fine grained blocks to customize the search space even
further. See the following example.
"""

import autokeras as ak

input_node = ak.TextInput()
output_node = ak.TextToIntSequence()(input_node)
output_node = ak.Embedding()(output_node)
# Use separable Conv layers in Keras.
output_node = ak.ConvBlock(separable=True)(output_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=1)
clf.fit(x_train, y_train, epochs=2)

"""
## Data Format
The AutoKeras TextClassifier is quite flexible for the data format.

For the text, the input data should be one-dimensional 
For the classification labels, AutoKeras accepts both plain labels, i.e. strings or
integers, and one-hot encoded encoded labels, i.e. vectors of 0s and 1s.

We also support using [tf.data.Dataset](
https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=stable) format for
the training data.
The labels have to be one-hot encoded for multi-class
classification to be wrapped into tensorflow Dataset.
Since the IMDB dataset is binary classification, it should not be one-hot encoded.
"""

import tensorflow as tf
train_set = tf.data.Dataset.from_tensor_slices(((x_train, ), (y_train, ))).batch(32)
test_set = tf.data.Dataset.from_tensor_slices(((x_test, ), (y_test, ))).batch(32)

clf = ak.TextClassifier(max_trials=3)
# Feed the tensorflow Dataset to the classifier.
clf.fit(train_set, epochs=2)
# Predict with the best model.
predicted_y = clf.predict(test_set)
# Evaluate the best model with testing data.
print(clf.evaluate(test_set))

"""
## Reference
[TextClassifier](/text_classifier),
[AutoModel](/auto_model/#automodel-class),
[TextBlock](/block/#textblock-class),
[TextToInteSequence](/preprocessor/#texttointsequence-class),
[Embedding](/block/#embedding-class),
[TextToNgramVector](/preprocessor/#texttongramvector-class),
[ConvBlock](/block/#convblock-class),
[TextInput](/node/#textinput-class),
[ClassificationHead](/head/#classificationhead-class).
"""
