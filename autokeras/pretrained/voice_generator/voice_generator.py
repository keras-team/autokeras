import lws
import os

import librosa
import numpy as np
import torch
from scipy import signal

from autokeras.constant import Constant
from autokeras.pretrained.base import Pretrained
from autokeras.pretrained.voice_generator.deepvoice3_pytorch import frontend, builder
from autokeras.utils import temp_path_generator, ensure_dir, get_device, download_file_from_google_drive


# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:

class Hparams:
    name = "deepvoice3"

    # Text:
    # [en  jp]
    frontend = 'en'

    # Replace words to its pronunciation with fixed probability.
    # e.g.  'hello' to 'HH AH0 L OW1'
    # [en  jp]
    # en: Word -> pronunciation using CMUDict
    # jp: Word -> pronounciation usnig MeCab
    # [0 ~ 1.0]: 0 means no replacement happens.
    replace_pronunciation_prob = 0.5

    # Convenient model builder
    # Definitions can be found at deepvoice3_pytorch/builder.py
    # deepvoice3: DeepVoice3 https://arxiv.org/abs/1710.07654
    builder = "deepvoice3"

    # Must be configured depends on the dataset and model you use
    n_speakers = 1
    speaker_embed_dim = 16

    # Audio:
    num_mels = 80
    fmin = 125
    fmax = 7600
    fft_size = 1024
    hop_size = 256
    sample_rate = 22050
    preemphasis = 0.97
    min_level_db = -100
    ref_level_db = 20
    # whether to rescale waveform or not.
    # Let x is an input waveform  rescaled waveform y is given by:
    # y = x / np.abs(x).max() * rescaling_max
    rescaling = False
    rescaling_max = 0.999
    # mel-spectrogram is normalized to [0  1] for each utterance and clipping may
    # happen depends on min_level_db and ref_level_db  causing clipping noise.
    # If False  assertion is added to ensure no clipping happens.
    allow_clipping_in_normalization = True

    # Model:
    downsample_step = 4  # must be 4 when builder="nyanko"
    outputs_per_step = 1  # must be 1 when builder="nyanko"
    embedding_weight_std = 0.1
    speaker_embedding_weight_std = 0.01
    padding_idx = 0
    # Maximum number of input text length
    # try setting larger value if you want to give very long text input
    max_positions = 512
    dropout = 1 - 0.95
    kernel_size = 3
    text_embed_dim = 128
    encoder_channels = 256
    decoder_channels = 256
    # Note: large converter channels requires significant computational cost
    converter_channels = 256
    query_position_rate = 1.0
    # can be computed by `compute_timestamp_ratio.py`.
    key_position_rate = 1.385  # 2.37 for jsut
    key_projection = False
    value_projection = False
    use_memory_mask = True
    trainable_positional_encodings = False
    freeze_embedding = False
    # If True  use decoder's internal representation for postnet inputs
    # otherwise use mel-spectrogram.
    use_decoder_state_for_postnet_input = True

    # Data loader
    pin_memory = True
    num_workers = 2  # Set it to 1 when in Windows (MemoryError  THAllocator.c 0x5)

    # Loss
    masked_loss_weight = 0.5  # (1-w)*loss + w * masked_loss
    priority_freq = 3000  # heuristic: priotrize [0 ~ priotiry_freq] for linear loss
    priority_freq_weight = 0.0  # (1-w)*linear_loss + w*priority_linear_loss
    # https://arxiv.org/pdf/1710.08969.pdf
    # Adding the divergence to the loss stabilizes training  expecially for
    # very deep (> 10 layers) networks.
    # Binary div loss seems has approx 10x scale compared to L1 loss  so I choose 0.1.
    binary_divergence_weight = 0.1  # set 0 to disable
    use_guided_attention = True
    guided_attention_sigma = 0.2

    # Training:
    batch_size = 16
    adam_beta1 = 0.5
    adam_beta2 = 0.9
    adam_eps = 1e-6
    amsgrad = False
    initial_learning_rate = 5e-4  # 0.001
    lr_schedule = "noam_learning_rate_decay"
    lr_schedule_kwargs = {}
    nepochs = 2000
    weight_decay = 0.0
    clip_thresh = 0.1

    # Save
    checkpoint_interval = 10000
    eval_interval = 10000
    save_optimizer_state = True

    # Eval:
    # this can be list for multple layers of attention
    # e.g.  [True  False  False  False  True]
    force_monotonic_attention = True
    # Attention constraint for incremental decoding
    window_ahead = 3
    # 0 tends to prevent word repretetion  but sometime causes skip words
    window_backward = 1
    power = 1.4  # Power to raise magnitudes to prior to phase retrieval

    # GC:
    # Forced garbage collection probability
    # Use only when MemoryError continues in Windows (Disabled by default)
    # gc_probability = 0.001

    # json_meta mode only
    # 0: "use all"
    # 1: "ignore only unmatched_alignment"
    # 2: "fully ignore recognition"
    ignore_recognition_level = 2
    # when dealing with non-dedicated speech dataset(e.g. movie excerpts)  setting min_text above 15 is desirable.
    # Can be adjusted by dataset.
    min_text = 20
    # if true  data without phoneme alignment file(.lab) will be ignored
    process_only_htk_aligned = False


fs = Hparams.sample_rate
global_step = 0
global_epoch = 0


def build_model():
    model = getattr(builder, Hparams.builder)(
        n_speakers=Hparams.n_speakers,
        speaker_embed_dim=Hparams.speaker_embed_dim,
        n_vocab=frontend.n_vocab,
        embed_dim=Hparams.text_embed_dim,
        mel_dim=Hparams.num_mels,
        linear_dim=Hparams.fft_size // 2 + 1,
        r=Hparams.outputs_per_step,
        padding_idx=Hparams.padding_idx,
        dropout=Hparams.dropout,
        kernel_size=Hparams.kernel_size,
        encoder_channels=Hparams.encoder_channels,
        decoder_channels=Hparams.decoder_channels,
        converter_channels=Hparams.converter_channels,
        use_memory_mask=Hparams.use_memory_mask,
        trainable_positional_encodings=Hparams.trainable_positional_encodings,
        force_monotonic_attention=Hparams.force_monotonic_attention,
        use_decoder_state_for_postnet_input=Hparams.use_decoder_state_for_postnet_input,
        max_positions=Hparams.max_positions,
        freeze_embedding=Hparams.freeze_embedding,
        window_ahead=Hparams.window_ahead,
        window_backward=Hparams.window_backward
    )
    return model


def inv_preemphasis(x, coef=Hparams.preemphasis):
    """Inverse operation of pre-emphasis

    Args:
        x (1d-array): Input signal.
        coef (float): Pre-emphasis coefficient.

    Returns:
        array: Output filtered signal.

    See also:
        :func:`preemphasis`
    """
    b = np.array([1.], x.dtype)
    a = np.array([1., -coef], x.dtype)
    return signal.lfilter(b, a, x)


def inv_spectrogram(spectrogram):
    """Converts spectrogram to waveform using librosa"""
    S = _db_to_amp(_denormalize(spectrogram) + Hparams.ref_level_db)  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** Hparams.power)
    y = processor.istft(D).astype(np.float32)
    return inv_preemphasis(y)


def _lws_processor():
    return lws.lws(Hparams.fft_size, Hparams.hop_size, mode="speech")


_mel_basis = None


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -Hparams.min_level_db) + Hparams.min_level_db


class VoiceGenerator(Pretrained):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = 0
        self.hop_length = 0
        self.sample_rate = Hparams.sample_rate
        self.hop_length = Hparams.hop_size

        self.model = self.load_checkpoint()
        self.model.to(self.device)

    @property
    def _google_drive_files(self):
        return Constant.VOICE_GENERATOR_MODELS

    def load_checkpoint(self):
        global global_step
        global global_epoch

        model = build_model()
        print("Load checkpoint from: {}".format(self.local_paths[0]))
        if self.device.startswith("cuda"):
            checkpoint = torch.load(self.local_paths[0])
        else:
            checkpoint = torch.load(self.local_paths[0], map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

        return model

    def predict(self, text, path=None):
        waveform, alignment, spectrogram, mel = self.tts(text)
        if path is None:
            AssertionError('Please provide the output file path.')
        librosa.output.write_wav(path, waveform, self.sample_rate)

    def tts(self, text, p=0, speaker_id=None, fast=True):
        """Convert text to speech waveform given a deepvoice3 model.

        Args:
            speaker_id:
            fast:
            text (str) : Input text to be synthesized
            p (float) : Replace word to pronounciation if p > 0. Default is 0.
        """
        self.model.eval()
        if fast:
            self.model.make_generation_fast_()

        sequence = np.array(frontend.text_to_sequence(text, p=p))
        sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(self.device)
        text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(self.device)
        speaker_ids = None if speaker_id is None else torch.LongTensor([speaker_id]).to(self.device)

        # Greedy decoding
        with torch.no_grad():
            mel_outputs, linear_outputs, alignments, done = self.model(
                sequence, text_positions=text_positions, speaker_ids=speaker_ids)

        linear_output = linear_outputs[0].cpu().data.numpy()
        spectrogram = _denormalize(linear_output)
        alignment = alignments[0].cpu().data.numpy()
        mel = mel_outputs[0].cpu().data.numpy()
        mel = _denormalize(mel)

        # Predicted audio signal
        waveform = inv_spectrogram(linear_output.T)

        return waveform, alignment, spectrogram, mel
