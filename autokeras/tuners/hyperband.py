import kerastuner

from autokeras.engine import tuner as tuner_module


class Hyperband(kerastuner.Hyperband, tuner_module.AutoTuner):
    """KerasTuner Hyperband with preprocessing layer tuning."""
    pass
