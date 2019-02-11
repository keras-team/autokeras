from autokeras.pretrained import VoiceRecognizer

if __name__ == '__main__':
    voice_recognizer = VoiceRecognizer()
    print(voice_recognizer.predict(audio_path="data/test.wav"))

