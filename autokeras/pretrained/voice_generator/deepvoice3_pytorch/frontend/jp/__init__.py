# coding: utf-8


import MeCab
import jaconv
from random import random

n_vocab = 0xffff

_eos = 1
_pad = 0
_tagger = None


def _yomi(mecab_result):
    tokens = []
    yomis = []
    for line in mecab_result.split("\n")[:-1]:
        s = line.split("\t")
        if len(s) == 1:
            break
        token, rest = s
        rest = rest.split(",")
        tokens.append(token)
        yomi = rest[7] if len(rest) > 7 else None
        yomi = None if yomi == "*" else yomi
        yomis.append(yomi)

    return tokens, yomis


def _mix_pronunciation(tokens, yomis, p):
    return "".join(
        yomis[idx] if yomis[idx] is not None and random() < p else tokens[idx]
        for idx in range(len(tokens)))


def mix_pronunciation(text, p):
    global _tagger
    if _tagger is None:
        _tagger = MeCab.Tagger("")
    tokens, yomis = _yomi(_tagger.parse(text))
    return _mix_pronunciation(tokens, yomis, p)


def add_punctuation(text):
    last = text[-1]
    if last not in [".", ",", "、", "。", "！", "？", "!", "?"]:
        text = text + "。"
    return text


def normalize_delimitor(text):
    text = text.replace(",", "、")
    text = text.replace(".", "。")
    text = text.replace("，", "、")
    text = text.replace("．", "。")
    return text


def text_to_sequence(text, p=0.0):
    for c in [" ", "　", "「", "」", "『", "』", "・", "【", "】",
              "（", "）", "(", ")"]:
        text = text.replace(c, "")
    text = text.replace("!", "！")
    text = text.replace("?", "？")

    text = normalize_delimitor(text)
    text = jaconv.normalize(text)
    if p > 0:
        text = mix_pronunciation(text, p)
    text = jaconv.hira2kata(text)
    text = add_punctuation(text)

    return [ord(c) for c in text] + [_eos]  # EOS


def sequence_to_text(seq):
    return "".join(chr(n) for n in seq)
