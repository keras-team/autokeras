from autokeras.pretrained import VoiceRecognizer
from autokeras.constant import Constant
import torchaudio
import scipy.signal
import librosa
import torch
import numpy as np


def load_audio(path):
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[0] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=0)  # multiple channels, average
    return sound


class SpectrogramParser:
    def __init__(self, audio_conf, normalize=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = scipy.signal.hamming
        self.normalize = normalize
        self.augment = augment
        self.noise_prob = audio_conf.get('noise_prob')

    def parse_audio(self, audio_path):
        y = load_audio(audio_path)

        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, _ = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect


if __name__ == '__main__':
    # First we need to parse the audio into tensor format

    # 1. initialize the parser as SpectrogramParser with audio_conf in Constant;
    parser = SpectrogramParser(Constant.VOICE_RECONGINIZER_AUDIO_CONF, normalize=True)
    # 2. given the audio path to the parser and parse the audio in the following way;
    spect = parser.parse_audio("data/test.wav").contiguous()
    voice_recognizer = VoiceRecognizer()
    print(voice_recognizer.predict(audio_data=spect))
