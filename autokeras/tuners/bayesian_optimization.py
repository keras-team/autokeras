import kerastuner

from autokeras.engine import tuner as tuner_module


class BayesianOptimization(kerastuner.BayesianOptimization, tuner_module.AutoTuner):
    """KerasTuner BayesianOptimization with preprocessing layer tuning."""
    pass
