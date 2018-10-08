import os
import re

import numpy as np
from keras import Input, Model
from keras.layers import Embedding
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import tensorflow as tf

from autokeras.constant import Constant
from autokeras.utils import download_file_with_extract
import GPUtil
from keras import backend as K

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


def tokenlize_text(max_num_words, max_seq_length, x_train):
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
    print("data readed and convert to %d length sequences" % max_seq_length)
    return x_train, word_index


def read_embedding_index(extract_path):
    embedding_index = {}
    f = open(os.path.join(extract_path, Constant.PRE_TRAIN_FILE_NAME))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    return embedding_index


def load_pretrain(path, word_index):
    print("loading pretrain weights...")
    file_path = os.path.join(path, Constant.FILE_PATH)
    extract_path = os.path.join(path, Constant.EXTRACT_PATH)
    download_pre_train(file_path=file_path, extract_path=extract_path)
    embedding_index = read_embedding_index(extract_path)
    print('Total %s word vectors embedded.' % len(embedding_index))

    # convert the pretrained embedding index to weights
    embedding_matrix = np.random.random((len(word_index) + 1, Constant.EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def processing(path, word_index, input_length, x_train):
    embedding_matrix = load_pretrain(path=path, word_index=word_index)

    # Get the first available GPU
    DEVICE_ID_LIST = GPUtil.getFirstAvailable()
    DEVICE_ID = DEVICE_ID_LIST[0]  # grab first element from list

    # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    device = '/gpu:0'
    with tf.device(device):
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
        print("generating preprocessing model...")
        embedding_layer = Embedding(len(word_index) + 1,
                                    Constant.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=input_length,
                                    trainable=False)

        sequence_input = Input(shape=(input_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        model = Model(sequence_input, embedded_sequences)
        print("converting text to vector...")
        x_train = model.predict(x_train)
        del model
    x_train = np.expand_dims(x_train, -1)
    return x_train


def text_preprocess(x_train, path):
    x_train = [clean_str(x) for x in x_train]
    x_train, word_index = tokenlize_text(max_seq_length=Constant.MAX_SEQUENCE_LENGTH,
                                         max_num_words=Constant.MAX_NB_WORDS,
                                         x_train=x_train)

    print("generating preprocessing model...")
    x_train = processing(path=path, word_index=word_index, input_length=Constant.MAX_SEQUENCE_LENGTH, x_train=x_train)

    return x_train
