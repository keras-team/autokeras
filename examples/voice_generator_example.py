from autokeras import VoiceGenerator
from autokeras.pretrained.voice_generator.hparams import Hparams
print(Hparams.builder)

voice_generator = VoiceGenerator()
text = "This is expensive, it costs me $300.2"
voice_generator.generate(text, path="save.wav")
