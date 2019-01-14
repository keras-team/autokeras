import lws

import numpy as np
from scipy import signal

from autokeras.pretrained.voice_generator.hparams import Hparams


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
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + Hparams.ref_level_db)  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** Hparams.power)
    y = processor.istft(D).astype(np.float32)
    return inv_preemphasis(y)


def _lws_processor():
    return lws.lws(Hparams.fft_size, Hparams.hop_size, mode="speech")


# Conversions:


_mel_basis = None


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -Hparams.min_level_db) + Hparams.min_level_db
