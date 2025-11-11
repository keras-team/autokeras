# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter

import keras
import numpy as np

from autokeras.engine import preprocessor


@keras.utils.register_keras_serializable(package="autokeras")
class AddOneDimension(preprocessor.Preprocessor):
    """Append one dimension of size one to the dataset shape."""

    def transform(self, dataset):
        return np.expand_dims(dataset, axis=-1)


@keras.utils.register_keras_serializable(package="autokeras")
class CastToInt32(preprocessor.Preprocessor):
    """Cast the dataset shape to int32."""

    def transform(self, dataset):
        return dataset.astype("int32")


@keras.utils.register_keras_serializable(package="autokeras")
class CastToString(preprocessor.Preprocessor):
    """Cast the dataset shape to string."""

    def transform(self, dataset):
        if np.issubdtype(dataset.dtype, np.bytes_):
            return np.array(
                [x.decode("utf-8", errors="ignore") for x in dataset]
            )
        else:
            return dataset.astype("str")


@keras.utils.register_keras_serializable(package="autokeras")
class TextTokenizer(preprocessor.Preprocessor):
    """Simple text tokenizer that converts strings to integer sequences."""

    def __init__(self, max_len=100, vocab=None, max_vocab=500, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.vocab = vocab
        self.max_vocab = max_vocab

    def fit(self, dataset):
        # Build vocab from unique words in the dataset
        unique_words = []
        for text in dataset:
            words = text.split()
            unique_words.extend(words)
        word_counts = Counter(unique_words)
        sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
        self.vocab = {
            word: idx + 1
            for idx, word in enumerate(sorted_words[: self.max_vocab])
        }  # Start from 1, 0 for padding

    def transform(self, dataset):
        # dataset is np.array of strings
        sequences = []
        for text in dataset:
            words = text.split()[: self.max_len]
            seq = [self.vocab.get(word, 0) for word in words]  # 0 for unknown
            # Pad with 0s
            seq += [0] * (self.max_len - len(seq))
            sequences.append(seq)
        return np.array(sequences, dtype=np.int32)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_len": self.max_len,
                "vocab": self.vocab,
                "max_vocab": self.max_vocab,
            }
        )
        return config
