import kerastuner

from autokeras.engine import tuner as tuner_module


class BayesianOptimization(tuner_module.AutoTuner, kerastuner.BayesianOptimization):
    """KerasTuner BayesianOptimization with preprocessing layer tuning."""
    pass
