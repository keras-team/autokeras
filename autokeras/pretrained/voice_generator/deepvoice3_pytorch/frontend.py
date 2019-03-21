# coding: utf-8
import os
import nltk
from autokeras.pretrained.voice_generator.deepvoice3_pytorch.text.symbols import symbols


N_VOCAB = len(symbols)
n_vocab = N_VOCAB

try:
    _ARPHABET = nltk.corpus.cmudict.dict()
except BaseException:
    nltk.download("cmudict")
    _ARPHABET = nltk.corpus.cmudict.dict()


def _maybe_get_arpabet(word, pro):
    try:
        phonemes = _ARPHABET[word][0]
        phonemes = " ".join(phonemes)
    except KeyError:
        return word

    return '{%s}' % phonemes if ord(os.urandom(1)) < pro else word


def mix_pronunciation(text, pro):
    text = ' '.join(_maybe_get_arpabet(word, pro) for word in text.split(' '))
    return text


def text_to_sequence(text, p=0.0):
    pro = p
    if pro >= 0:
        text = mix_pronunciation(text, pro)
    from autokeras.pretrained.voice_generator.deepvoice3_pytorch.text.text import text_to_sequence
    text = text_to_sequence(text, ["english_cleaners"])
    return text
