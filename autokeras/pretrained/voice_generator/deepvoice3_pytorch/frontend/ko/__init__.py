# coding: utf-8


from random import random

n_vocab = 0xffff

_eos = 1
_pad = 0
_tagger = None


def text_to_sequence(text, p=0.0):
    return [ord(c) for c in text] + [_eos]  # EOS

def sequence_to_text(seq):
    return "".join(chr(n) for n in seq)
