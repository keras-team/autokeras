import kerastuner

from autokeras.engine import tuner as tuner_module


class Hyperband(kerastuner.Hyperband, tuner_module.AutoTuner):
    """KerasTuner Hyperband with preprocessing layer tuning."""

    def __init__(self, max_epochs=1000, max_trials=100, *args, **kwargs):
        super().__init__(max_epochs=max_epochs, *args, **kwargs)
        self.oracle.max_trials = max_trials
