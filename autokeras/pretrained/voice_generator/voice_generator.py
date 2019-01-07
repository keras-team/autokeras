import os

from autokeras.pretrained.base import Pretrained
from autokeras.constant import Constant
from autokeras.utils import temp_path_generator, ensure_dir
import librosa

from autokeras.pretrained.voice_generator import hparams
from autokeras.pretrained.voice_generator import synthesis
from autokeras.pretrained.voice_generator import model_helper
from autokeras.pretrained.voice_generator.deepvoice3_pytorch import frontend
from autokeras.pretrained.voice_generator.model_helper import build_model, load_checkpoint
from autokeras.pretrained.voice_generator.synthesis import tts

from autokeras.pretrained.voice_generator.google_drive_download import GoogleDriveDownloader as gdd

synthesis._frontend = frontend
model_helper._frontend = frontend


class VoiceGenerator(Pretrained):
    def __init__(self, model_path=None):
        super(VoiceGenerator, self).__init__()
        if model_path is None:
            model_path = temp_path_generator()
        self.model_path = model_path
        ensure_dir(self.model_path)
        self.checkpoint_path = os.path.join(self.model_path, Constant.PRE_TRAIN_VOICE_GENERATOR_MODEL_NAME)
        self.sample_rate = 0
        self.hop_length = 0
        self.load()

    def load(self):
        self._maybe_download()
        self.sample_rate = hparams.Hparams.sample_rate
        self.hop_length = hparams.Hparams.hop_size
        model = build_model()

        self.model = load_checkpoint(self.checkpoint_path, model)

    def _maybe_download(self, overwrite=True):
        # For files in dropbox or google drive, cannot directly use request to download
        # This can be changed directly use download_file method when the file is stored in server
        if not os.path.exists(self.checkpoint_path) or overwrite:
            checkpoint_google_id = Constant.PRE_TRAIN_VOICE_GENERATOR_MODEL_GOOGLE_DRIVE_ID
            gdd.download_file_from_google_drive(file_id=checkpoint_google_id, dest_path=self.checkpoint_path, overwrite=overwrite)

    def generate(self, text, path=None):
        waveform, alignment, spectrogram, mel = tts(self.model, text)
        if path is None:
            path = Constant.PRE_TRAIN_VOICE_GENERATOR_SAVE_FILE_DEFAULT_NAME
        librosa.output.write_wav(path, waveform, self.sample_rate)

    def predict(self, x_predict):
        pass
