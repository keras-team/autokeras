from autokeras.pretrained import VoiceGenerator
import os

temp_dir = "test"
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)
if __name__ == '__main__':
    voice_generator = VoiceGenerator()
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
        save_name = os.path.join(temp_dir, save_name)
        voice_generator.predict(text, save_name)
