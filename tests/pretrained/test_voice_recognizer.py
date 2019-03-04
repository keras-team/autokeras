import torch

from autokeras.pretrained import VoiceRecognizer


def test_voice_generator():
    spect2 = torch.rand(161, 131)
    voice_recognizer = VoiceRecognizer()
    print(voice_recognizer.predict(audio_data=spect2))


def test_voice_generator_none_type_error():
    voice_recognizer = VoiceRecognizer()
    try:
        print(voice_recognizer.predict(audio_data=None))
    except TypeError:
        pass
