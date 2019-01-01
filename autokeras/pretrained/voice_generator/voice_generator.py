import os

from autokeras.pretrained.base import Pretrained
from autokeras.constant import Constant
from autokeras.utils import temp_path_generator
import librosa

from autokeras.pretrained.voice_generator import hparams
from autokeras.pretrained.voice_generator import synthesis
from autokeras.pretrained.voice_generator import model_helper
from autokeras.pretrained.voice_generator.deepvoice3_pytorch import frontend
from autokeras.pretrained.voice_generator.model_helper import build_model, load_checkpoint
from autokeras.pretrained.voice_generator.synthesis import tts

synthesis._frontend = getattr(frontend, "en")
model_helper._frontend = getattr(frontend, "en")


class VoiceGenerator(Pretrained):
    def __init__(self, model_path=None):
        super(VoiceGenerator, self).__init__()
        if model_path is None:
            model_path = temp_path_generator()
        self.model_path = model_path
        self.checkpoint_path = os.path.join(self.model_path, Constant.PRE_TRAIN_VOICE_GENERATOR_MODEL_NAME)
        self.sample_rate = 0
        self.hop_length = 0
        self.load()

    def load(self):
        self._maybe_download()
        self.sample_rate = hparams.hparams.sample_rate
        self.hop_length = hparams.hparams.hop_size
        model = build_model()

        self.model = load_checkpoint(self.checkpoint_path, model)

    def _maybe_download(self):
        # For files in dropbox or google drive, cannot directly use request to download
        # This can be changed directly use download_file method when the file is stored in server
        if not os.path.exists(self.checkpoint_path):
            print("Downloading " + self.checkpoint_path + " from " + Constant.PRE_TRAIN_VOICE_GENERATOR_MODEL_LINK)
            checkpoint_link = Constant.PRE_TRAIN_VOICE_GENERATOR_MODEL_LINK
            current_path = os.getcwd()
            os.chdir(self.model_path)
            cmd = "curl -O -L " + checkpoint_link
            os.system(cmd)
            os.chdir(current_path)

    def generate(self, text, path=None):
        waveform, alignment, spectrogram, mel = tts(self.model, text)
        if path is None:
            path = Constant.PRE_TRAIN_VOICE_GENERATOR_SAVE_FILE_DEFAULT_NAME
        librosa.output.write_wav(path, waveform, self.sample_rate)

    def predict(self, x_predict):
        pass
