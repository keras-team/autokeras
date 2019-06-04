import random
import numpy as np

from autokeras.constant import Constant
from autokeras.hypermodel.hypermodel import HyperModel, DefaultHyperModel
import autokeras.hyperparameters as hp_module
from autokeras.layer_utils import format_inputs


class Tuner(object):

    def __init__(self,
                 hypermodel,
                 objective=None,
                 optimizer=None,
                 loss=None,
                 metrics=None,
                 reparameterization=None,
                 tune_rest=True,
                 static_values=None,
                 allow_new_parameters=True,
                 **kwargs):
        self.allow_new_parameters = allow_new_parameters
        self.tune_rest = tune_rest
        if not reparameterization:
            self.hyperparameters = hp_module.HyperParameters()
        else:
            self.hyperparameters = hp_module.HyperParameters.from_config(
                reparameterization.get_config())
        if static_values:
            for name, value in static_values.items():
                print(name, value)
                self.hyperparameters.Choice(name, [value], default=value)

        if isinstance(hypermodel, HyperModel):
            self.hypermodel = hypermodel
        else:
            if not callable(hypermodel):
                raise ValueError(
                    'The `hypermodel` argument should be either '
                    'a callable with signature `build(hp)` returning a model, '
                    'or an instance of `HyperModel`.')
            self.hypermodel = DefaultHyperModel(hypermodel)
        if objective is not None:
            self.objective = objective
        elif metrics:
            self.objective = format_inputs(metrics, 'Tuner objective')[0]
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics


def convert_metric_to_higher_better(metric_value, metric):
    if metric in Constant.LOWER_BETTER:
        return 1 - metric_value
    return metric_value


class SequentialRandomSearch(Tuner):

    def __init__(self, hypermodel, **kwargs):
        super(SequentialRandomSearch, self).__init__(hypermodel, **kwargs)
        self.best_hp = None
        self.best_feedback = -float('inf')
        # TODO: clear the memory of the models and save the best model in disk instead of memory.
        self.best_model = None

    def search(self, trials, **kwargs):
        for _ in range(trials):
            if self.tune_rest:
                # In this case, append to the space,
                # so pass the internal hp object to `build`
                hp = self.hyperparameters
            else:
                # In this case, never append to the space
                # so work from a copy of the internal hp object
                hp = hp_module.HyperParameters.from_config(
                    self.hyperparameters.get_config())
            hp = self._populate_hyperparameter_values(hp)
            model, feedback = self._run(hp, fit_kwargs=kwargs)
            if feedback > self.best_feedback:
                self.best_feedback = feedback
                self.best_hp = hp
                self.best_model = model

    def _run(self, hyperparameters, fit_kwargs):
        # Build a model instance.
        model = self.hypermodel.build(hyperparameters)

        # Optionally disallow hyperparameters defined on the fly.
        old_space = hyperparameters.space[:]
        new_space = hyperparameters.space[:]
        if not self.allow_new_parameters and set(old_space) != set(new_space):
            diff = set(new_space) - set(old_space)
            raise RuntimeError(
                'The hypermodel has requested a parameter that was not part '
                'of `hyperparameters`, '
                'yet `allow_new_parameters` is set to False. '
                'The unknown parameters are: {diff}'.format(diff=diff))

        # Optional recompile
        if not model.optimizer:
            model.compile()
        elif self.optimizer or self.loss or self.metrics:
            compile_kwargs = {
                'optimizer': model.optimizer,
                'loss': model.loss,
                'metrics': model.metrics,
            }
            if self.loss:
                compile_kwargs['loss'] = self.loss
            if self.optimizer:
                compile_kwargs['optimizer'] = self.optimizer
            if self.metrics:
                compile_kwargs['metrics'] = self.metrics
            model.compile()

        # Train model
        # TODO: reporting presumably done with a callback, record the hp and performances
        history = model.fit(**fit_kwargs)

        metric_name = model.metrics_names[1]
        feedback = history.history['val_' + metric_name][-1]
        return model, convert_metric_to_higher_better(feedback, metric_name)

    def _populate_hyperparameter_values(self, hyperparameters):
        for p in hyperparameters.space:
            hyperparameters.values[p.name] = self._sample_parameter(p)
        return hyperparameters

    def _sample_parameter(self, parameter):
        if isinstance(parameter, hp_module.Choice):
            return random.choice(parameter.values)
        elif isinstance(parameter, hp_module.Range):
            if parameter.step:
                value = random.choice(list(np.arange(
                    parameter.min_value, parameter.max_value, parameter.step)))
                if (isinstance(parameter.min_value, int) and
                        isinstance(parameter.max_value, int) and
                        isinstance(parameter.step, int)):
                    return int(value)
                return value
            else:
                value = np.random.uniform(
                    parameter.min_value, parameter.max_value)
                if (isinstance(parameter.min_value, int) and
                        isinstance(parameter.max_value, int)):
                    return int(value)
                return value
        return None
