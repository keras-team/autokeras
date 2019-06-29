import kerastuner


class AutoTuner(kerastuner.Tuner):

    def run_trial(self, trial, hp, fit_args, fit_kwargs):
        x = self.hypermodel.preprocess(hp, *fit_args, **fit_kwargs)

        if 'x' in fit_kwargs:
            fit_kwargs['x'] = x
        else:
            fit_args[0] = x

        super(AutoTuner, self).run_trial(trial, hp, fit_args, fit_kwargs)


class RandomSearch(AutoTuner, kerastuner.RandomSearch):
    pass
