from autokeras import TextClassifier
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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


if __name__ == '__main__':
    data_train = pd.read_csv('labeledTrainData.tsv', sep='\t')
    print(data_train.shape)
    texts = []
    y_train = []
    for idx in range(data_train.review.shape[0]):
        texts.append(clean_str(data_train.review[idx]))
        y_train.append(data_train.sentiment[idx])

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    clf = TextClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    print(y * 100)