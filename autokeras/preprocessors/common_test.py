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

import numpy as np

from autokeras import test_utils
from autokeras.preprocessors import common


def test_cast_to_int32_return_int32():
    x = test_utils.generate_one_hot_labels(100, 10)
    x = x.astype("uint8")
    x = common.CastToInt32().transform(x)
    assert x.dtype == "int32"


def test_cast_to_string_with_bytes():
    x = np.array([b"hello", b"world"])
    result = common.CastToString().transform(x)
    assert result.dtype.kind in ["U", "S"]  # Unicode or byte string
    assert result[0] == "hello"
    assert result[1] == "world"


def test_cast_to_string_with_strings():
    x = np.array(["hello", "world"])
    result = common.CastToString().transform(x)
    assert result.dtype.kind in ["U", "S"]
    assert result[0] == "hello"
    assert result[1] == "world"


def test_text_tokenizer_vocab_limit():
    x = np.array(["word1 word2 word3", "word1 word4 word5"])
    tokenizer = common.TextTokenizer(max_vocab=2)
    tokenizer.fit(x)
    assert len(tokenizer.vocab) <= 3  # 2 words + 1 for unknown (0 is padding)
    # word1 should be most frequent
    assert "word1" in tokenizer.vocab
    assert tokenizer.vocab["word1"] == 1


def test_text_tokenizer_transform():
    x = np.array(["hello world", "hello"])
    tokenizer = common.TextTokenizer(max_vocab=10)
    tokenizer.fit(x)
    result = tokenizer.transform(x)
    assert result.shape == (2, 100)  # max_len=100
    assert result.dtype == np.int32
    assert result[0][0] == tokenizer.vocab.get("hello", 0)
