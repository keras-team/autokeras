import kerastuner

from autokeras.engine import tuner as tuner_module


class Hyperband(tuner_module.AutoTuner, kerastuner.Hyperband):
    """KerasTuner Hyperband with preprocessing layer tuning."""
    pass
