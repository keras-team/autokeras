import pytest
from autokeras.tuner import AutoTuner
from tensorflow.keras.callbacks import EarlyStopping


@pytest.fixture(scope='module')
def test_add_earlystopping_callback():
    callbacks = []
    callbacks = AutoTuner.add_earlystopping_callback(callbacks)
    assert len(callbacks) == 1
    assert callbacks[0].__class__.__name__ == 'EarlyStopping'

    callbacks = [EarlyStopping(patience=2)]
    callbacks = AutoTuner.add_earlystopping_callback(callbacks)
    assert len(callbacks) == 1
    assert callbacks[0].__class__.__name__ == 'EarlyStopping'
