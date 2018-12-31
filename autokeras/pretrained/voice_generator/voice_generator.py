from autokeras.pretrained.base import Pretrained
from autokeras.constant import Constant
from autokeras.utils import temp_path_generator, download_file
import librosa

import hparams
import synthesis
import model_helper
from deepvoice3_pytorch import frontend
from model_helper import build_model
from model_helper import load_checkpoint
from synthesis import tts

synthesis._frontend = getattr(frontend, "en")
model_helper._frontend = getattr(frontend, "en")


class VoiceGenerator(Pretrained):
    def __init__(self, model_path=None):
        super(VoiceGenerator, self).__init__()
        if model_path is None:
            model_path = temp_path_generator()
        self.model_path = model_path
        self.hyperparameter_path = self.model_path + Constant.PRE_TRAIN_VOICE_GENERATOR_MODEL_NAME
        self.checkpoint_path = self.model_path + Constant.PRE_TRAIN_VOICE_GENERATOR_HYPERPARAMETER_NAME
        self.sample_rate = 0
        self.hop_length = 0
        self.load()

    def load(self):
        self._maybe_download()
        with open(self.hyperparameter_path) as f:
            hparams.hparams.parse_json(f.read())
        self.sample_rate = hparams.hparams.sample_rate
        self.hop_length = hparams.hparams.hop_size
        model = build_model()
        self.model = load_checkpoint(self.checkpoint_path, model)

    def _maybe_download(self):
        checkpoint_link = Constant.PRE_TRAIN_VOICE_GENERATOR_MODEL_LINK
        download_file(checkpoint_link, self.checkpoint_path)

        hyperparameter_link = Constant.PRE_TRAIN_VOICE_GENERATOR_HYPERPARAMETER_LINK
        download_file(hyperparameter_link, self.hyperparameter_path)

    def generate(self, text, path=None):
        waveform, alignment, spectrogram, mel = tts(self.model, text)
        if path is None:
            path = Constant.PRE_TRAIN_VOICE_GENERATOR_SAVE_FILE_NAME
        librosa.output.write_wav(path, waveform, self.sample_rate)

    def predict(self, x_predict):
        pass
