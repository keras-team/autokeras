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
        self.hyperparameter_path = self.model_path + "20180505_deepvoice3_ljspeech.json"
        self.checkpoint_path = self.model_path + "20180505_deepvoice3_checkpoint_step000640000.pth"
        self.load()

    def load(self):
        self._maybe_download()

        pass

    def _maybe_download(self):
        checkpoint_link = Constant.PRE_TRAIN_VOICE_GENERATOR_MODEL_LINK
        download_file(checkpoint_link, self.checkpoint_path)

        hyperparameter_link = Constant.PRE_TRAIN_VOICE_GENERATOR_HYPERPARAMETER_LINK
        download_file(hyperparameter_link, self.hyperparameter_path)

    def generate(self, text, path=None):
        pass

    def predict(self, x_predict):
        pass
