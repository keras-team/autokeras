import kerastuner

from autokeras.engine import tuner as tuner_module


class RandomSearch(kerastuner.RandomSearch, tuner_module.AutoTuner):
    """KerasTuner RandomSearch with preprocessing layer tuning."""
    pass
