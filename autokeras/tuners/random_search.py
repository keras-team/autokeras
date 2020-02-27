import kerastuner

from autokeras.engine import tuner as tuner_module


class RandomSearch(tuner_module.AutoTuner, kerastuner.RandomSearch):
    """KerasTuner RandomSearch with preprocessing layer tuning."""
    pass
