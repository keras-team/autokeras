import os
import re

import numpy as np
from keras import Input, Model
from keras.layers import Embedding
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from autokeras.constant import Constant
from autokeras.utils import download_file_with_extract


def download_pre_train(file_path, extract_path):
    file_link = Constant.PRE_TRAIN_FILE_LINK
    print("try downloading pre train weights from link %s" % file_link)
    download_file_with_extract(file_link, file_path=file_path, extract_path=extract_path)


def clean_str(string):
    """
    Tokenization/string cleaning for all string.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def tokenlize_text(max_num_words, max_seq_length, x_train, y_train):
    """Tokenlize text class.

    Vectorize a text corpus by transform each text in texts to a sequence of integers.

    Attributes:
        max_num_words: int, max number of words in the dictionary
        max_seq_length: int, the length of each text sequence, padding if shorter, trim is longer
        x_train: list contains text data
        y_train: list contains label data
    """
    print("tokenlizing texts...")
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(x_train)
    sequences = tokenizer.texts_to_sequences(x_train)
    word_index = tokenizer.word_index
    x_train = pad_sequences(sequences, maxlen=max_seq_length)
    y_train = to_categorical(np.asarray(y_train))
    print("data readed and convert to %d length sequences" % max_seq_length)
    return x_train, y_train, word_index


def load_pretrain(path, word_index):
    print("loading pretrain weights...")
    file_path = os.path.join(path, Constant.FILE_PATH)
    extract_path = os.path.join(path, Constant.EXTRACT_PATH)
    download_pre_train(file_path=file_path, extract_path=extract_path)
    embeddings_index = {}
    f = open(os.path.join(extract_path, Constant.PRE_TRAIN_FILE_NAME))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors embedded.' % len(embeddings_index))

    # convert the pretrained embedding index to weights
    embedding_matrix = np.random.random((len(word_index) + 1, Constant.EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def preprocessing_model(path, word_index, input_length):
    embedding_matrix = load_pretrain(path=path, word_index=word_index)
    print("generating preprocessing model...")
    embedding_layer = Embedding(len(word_index) + 1,
                                Constant.EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)

    sequence_input = Input(shape=(input_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    model = Model(sequence_input, embedded_sequences)
    return model


def text_preprocess(x_train, y_train, path):
    x_train = [clean_str(x) for x in x_train]
    x_train, y_train, word_index = tokenlize_text(max_seq_length=Constant.MAX_SEQUENCE_LENGTH,
                                                  max_num_words=Constant.MAX_NB_WORDS,
                                                  x_train=x_train, y_train=y_train)

    print("generating preprocessing model...")
    model = preprocessing_model(path=path, word_index=word_index, input_length=Constant.MAX_SEQUENCE_LENGTH)

    print("converting text to vector...")
    x_train = model.predict(x_train)
    del model
    x_train = np.expand_dims(x_train, -1)

    return x_train, y_train
