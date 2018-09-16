import os
import re

import numpy as np
import pandas as pd
from keras.layers import Embedding, Input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
import sys
from autokeras import TextClassifier, ImageClassifier

## Example dataset can be download here:
# wget https://www.kaggle.com/c/word2vec-nlp-tutorial/download/labeledTrainData.tsv

# Example embedding pretrain can be download here:
# wget http://nlp.stanford.edu/data/glove.6B.zip


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def read_csv(max_num_words, max_seq_length, file_path=None, file_name=None):
    """The image classifier class.

    It is used for image classification. It searches convolutional neural network architectures
    for the best configuration for the dataset.

    Attributes:
        max_num_words: max number of words in the dictionary
        max_seq_length: the length of each text sequence, padding if shorter, trim is longer
        file_path: csv file path
        file_name: csv file name
    """

    print("reading data...")
    if file_path is not None:
        file_dir = os.path.join(file_path, file_name)
    else:
        file_dir = file_name
    data_train = pd.read_csv(file_dir, sep='\t')
    print(data_train.shape)

    texts = []
    y_train = []
    for idx in range(data_train.review.shape[0]):
        # Modify this according to each different dataset
        texts.append(clean_str(data_train.review[idx]))
        y_train.append(data_train.sentiment[idx])

    tokenizer = Tokenizer(nb_words=max_num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    x_train = pad_sequences(sequences, maxlen=max_seq_length)
    y_train = to_categorical(np.asarray(y_train))
    print("data readed and convert to %d length sequences" % max_seq_length)
    return x_train, y_train, word_index


def load_pretrain(pretrain_dir, pretrain_file_name, word_index):
    embeddings_index = {}
    f = open(os.path.join(pretrain_dir, pretrain_file_name))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors embedded.' % len(embeddings_index))

    # convert the pretrained embedding index to weights
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def preprocessing_model(embedding_matrix, word_size, input_length):
    print("generating preprocessing model...")
    embedding_layer = Embedding(word_size + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=True)

    sequence_input = Input(shape=(input_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    model = Model(sequence_input, embedded_sequences)
    return model


if __name__ == '__main__':
    x_train, y_train, word_index = read_csv(MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, file_name="labeledTrainData.tsv")
    embedding_matrix = load_pretrain("glove_embedding", "glove.6B.100d.txt", word_index)
    pre_model = preprocessing_model(embedding_matrix, len(word_index), MAX_SEQUENCE_LENGTH)
    print("preprocessing data...")
    processed_x_train = pre_model.predict(x_train)
    processed_x_train = np.expand_dims(processed_x_train, -1)
    del pre_model
    print(processed_x_train.shape)
    print(sys.getsizeof(processed_x_train))

    clf = TextClassifier(verbose=True)
    clf.fit(processed_x_train, y_train, time_limit=12 * 60 * 60)
