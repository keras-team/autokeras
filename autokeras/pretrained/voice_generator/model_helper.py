"""Trainining script for seq2seq text-to-speech synthesis model.
"""

from warnings import warn

import torch
import torch.backends.cudnn as cudnn

# The deepvoice3 model
from deepvoice3_pytorch import builder
from hparams import hparams

fs = hparams.sample_rate

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

_frontend = None  # to be set later


def build_model():
    model = getattr(builder, hparams.builder)(
        n_speakers=hparams.n_speakers,
        speaker_embed_dim=hparams.speaker_embed_dim,
        n_vocab=_frontend.n_vocab,
        embed_dim=hparams.text_embed_dim,
        mel_dim=hparams.num_mels,
        linear_dim=hparams.fft_size // 2 + 1,
        r=hparams.outputs_per_step,
        downsample_step=hparams.downsample_step,
        padding_idx=hparams.padding_idx,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        encoder_channels=hparams.encoder_channels,
        decoder_channels=hparams.decoder_channels,
        converter_channels=hparams.converter_channels,
        use_memory_mask=hparams.use_memory_mask,
        trainable_positional_encodings=hparams.trainable_positional_encodings,
        force_monotonic_attention=hparams.force_monotonic_attention,
        use_decoder_state_for_postnet_input=hparams.use_decoder_state_for_postnet_input,
        max_positions=hparams.max_positions,
        speaker_embedding_weight_std=hparams.speaker_embedding_weight_std,
        freeze_embedding=hparams.freeze_embedding,
        window_ahead=hparams.window_ahead,
        window_backward=hparams.window_backward,
        key_projection=hparams.key_projection,
        value_projection=hparams.value_projection,
    )
    return model


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

