# coding: utf-8

"""Text processing frontend

All frontend module should have the following functions:

- text_to_sequence(text, p)
- sequence_to_text(sequence)

and the property:

- n_vocab

"""
from autokeras.pretrained.voice_generator.deepvoice3_pytorch.frontend import en

# optinoal Japanese frontend
try:
    from autokeras.pretrained.voice_generator.deepvoice3_pytorch.frontend import jp
except ImportError:
    jp = None

try:
    from autokeras.pretrained.voice_generator.deepvoice3_pytorch.frontend import ko
except ImportError:
    ko = None

