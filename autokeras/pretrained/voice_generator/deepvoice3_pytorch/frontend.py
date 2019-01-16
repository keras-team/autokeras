# coding: utf-8
from autokeras.pretrained.voice_generator.deepvoice3_pytorch.text.symbols import symbols

import nltk
from random import random

n_vocab = len(symbols)

try:
    _arphabet = nltk.corpus.cmudict.dict()
except:
    nltk.download("cmudict")
    _arphabet = nltk.corpus.cmudict.dict()


def _maybe_get_arpabet(word, p):
    try:
        phonemes = _arphabet[word][0]
        phonemes = " ".join(phonemes)
    except KeyError:
        return word

    return '{%s}' % phonemes if random() < p else word


def mix_pronunciation(text, p):
    text = ' '.join(_maybe_get_arpabet(word, p) for word in text.split(' '))
    return text


def text_to_sequence(text, p=0.0):
    if p >= 0:
        text = mix_pronunciation(text, p)
    from autokeras.pretrained.voice_generator.deepvoice3_pytorch.text.text import text_to_sequence
    text = text_to_sequence(text, ["english_cleaners"])
    return text
