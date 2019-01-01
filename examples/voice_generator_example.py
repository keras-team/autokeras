from autokeras.pretrained.voice_generator.voice_generator import VoiceGenerator

voice_generator = VoiceGenerator()
text = "This is expensive, it costs me $300.2"
voice_generator.generate(text, path="save.wav")
