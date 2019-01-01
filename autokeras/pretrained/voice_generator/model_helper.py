"""Trainining script for seq2seq text-to-speech synthesis model.
"""

from warnings import warn

import torch
import torch.backends.cudnn as cudnn

# The deepvoice3 model
from autokeras.pretrained.voice_generator.deepvoice3_pytorch import builder
from autokeras.pretrained.voice_generator.hparams import Hparams

fs = Hparams.sample_rate

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

_frontend = None  # to be set later


def build_model():
    model = getattr(builder, Hparams.builder)(
        n_speakers=Hparams.n_speakers,
        speaker_embed_dim=Hparams.speaker_embed_dim,
        n_vocab=_frontend.n_vocab,
        embed_dim=Hparams.text_embed_dim,
        mel_dim=Hparams.num_mels,
        linear_dim=Hparams.fft_size // 2 + 1,
        r=Hparams.outputs_per_step,
        downsample_step=Hparams.downsample_step,
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
        speaker_embedding_weight_std=Hparams.speaker_embedding_weight_std,
        freeze_embedding=Hparams.freeze_embedding,
        window_ahead=Hparams.window_ahead,
        window_backward=Hparams.window_backward,
        key_projection=Hparams.key_projection,
        value_projection=Hparams.value_projection,
    )
    return model


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

