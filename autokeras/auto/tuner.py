import kerastuner


class AutoTuner(kerastuner.Tuner):

    def run_trial(self, trial, hp, fit_args, fit_kwargs):
        x, y = self.hypermodel.preprocess(*fit_args, **fit_kwargs)

        if 'x' in fit_kwargs:
            fit_kwargs['x'] = x
            fit_kwargs['y'] = y
        else:
            fit_args[0] = x
            if 'y' in fit_kwargs:
                fit_kwargs['y'] = y
            else:
                fit_args[1] = y

        super(AutoTuner, self).run_trial(trial, hp, fit_args, fit_kwargs)


class RandomSearch(AutoTuner, kerastuner.RandomSearch):
    pass
