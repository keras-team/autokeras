from autokeras.pretrained import VoiceGenerator
from tests.common import TEST_TEMP_DIR, clean_dir
import os


def test_voice_generator():
    voice_generator = VoiceGenerator()
    clean_dir(TEST_TEMP_DIR)
    texts = [
        "Generative adversarial network or variational auto-encoder.",
        "The tuition of the coming semster is 6300 dollars.",
        "The tuition of the coming semster is 6350 dollars.",
        "Turn left on {HH AW1 S S T AH0 N} Street.",
        "This is expensive, it costs me $300.2",
        "This is expensive, it costs me $300",
        "This is cheap, it only costs me $.2",
        "Today he won the 1st prize of the competition",
        "The approximation of pi is 3.14",
    ]

    for idx, text in enumerate(texts):
        save_name = "test_" + str(idx) + ".wav"
        save_name = os.path.join(TEST_TEMP_DIR, save_name)
        voice_generator.predict(text, path=save_name)
    clean_dir(TEST_TEMP_DIR)
