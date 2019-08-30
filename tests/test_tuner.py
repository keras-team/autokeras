import pytest
from autokeras.tuner import AutoTuner
from tensorflow.keras.callbacks import EarlyStopping


def test_add_earlystopping_callback():
    callbacks = []
    callbacks = AutoTuner.add_earlystopping_callback(callbacks)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], EarlyStopping)


def test_not_add_earlystopping_callback():
    callbacks = [EarlyStopping(patience=2)]
    callbacks = AutoTuner.add_earlystopping_callback(callbacks)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], EarlyStopping)
