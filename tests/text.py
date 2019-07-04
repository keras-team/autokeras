import numpy as np

from tensorflow.python.keras.preprocessing.text import Tokenizer

texts = ['The cat sat on the mat.',
         'The dog sat on the log.',
         'Dogs and cats living together.']
tokenizer = Tokenizer(num_words=10)
tokenizer.fit_on_texts(texts)

sequences = []
for seq in tokenizer.texts_to_sequences_generator(texts):
    sequences.append(seq)
assert np.max(np.max(sequences)) < 10
assert np.min(np.min(sequences)) == 1

print(sequences)
tokenizer.fit_on_sequences(sequences)

for mode in ['binary', 'count', 'tfidf', 'freq']:
    matrix = tokenizer.texts_to_matrix(texts, mode)
    print(matrix)

matrix = tokenizer.texts_to_sequences(texts)
print(matrix)

matrix = tokenizer.texts_to_sequences(texts[0])
print(matrix)
