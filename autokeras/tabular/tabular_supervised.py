import os
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
import pickle
import numpy as np
from os.path import isfile
import random
import time

from autokeras.supervised import Supervised


def search(lgbm, search_space, search_iter, n_estimators, x, y):
    if 'n_estimators' in search_space:
        del search_space['n_estimators']
    params = {
        'boosting_type': ['gbdt'],
        'min_child_weight': [5],
        'min_split_gain': [1.0],
        'subsample': [0.8],
        'colsample_bytree': [0.6],
        'max_depth': [10],
        'n_estimators': n_estimators,
        'num_leaves': [70],
        'learning_rate': [0.04],
    }

    params.update(search_space)
    print(params)
    folds = 3
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    random_search = RandomizedSearchCV(lgbm, param_distributions=params, n_iter=search_iter, scoring='roc_auc',
                                       n_jobs=1, cv=skf, verbose=0, random_state=1001)

    random_search.fit(x, y)

    return random_search.best_estimator_, random_search.best_params_


class TabularSupervised(Supervised):
    def __init__(self):
        """
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        """
        self.is_trained = False
        self.clf = None
        self.save_filename = None
        self.objective = 'multiclass'
        self.lgbm = LGBMClassifier(silent=False,
                                   verbose=-1,
                                   n_jobs=1,
                                   objective=self.objective)

    def fit(self, x, y, x_test=None, y_test=None, time_limit=None):
        """
        This function should train the model parameters.

        Args:
            x: A numpy.ndarray instance containing the training data.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs X and y are numpy arrays.
        If fit is called multiple times on incremental data (train, test1, test2, etc.)
        you should warm-start your training from the pre-trained model. Past data will
        NOT be available for re-training.
        """

        y0 = np.where(y == 0)[0]
        y1 = np.where(y == 1)[0]
        N0 = len(y0)
        N1 = len(y1)

        if x.shape[0] > 600:
            grid_train_perentage = 0.1
        else:
            grid_train_perentage = 1
        grid_N = int(x.shape[0] * grid_train_perentage)
        grid_half_N = int(grid_N * 0.5)
        if N0 > N1:
            if N1 <= grid_half_N:
                idx1 = list(y1)
                idx0 = random.sample(list(y0), grid_N - N1)
            else:
                idx1 = random.sample(list(y1), grid_half_N)
                idx0 = random.sample(list(y0), grid_half_N)
        else:
            if N0 <= grid_half_N:
                idx0 = list(y0)
                idx1 = random.sample(list(y1), grid_N - N0)
            else:
                idx1 = random.sample(list(y1), grid_half_N)
                idx0 = random.sample(list(y0), grid_half_N)

        grid_train_samples = sorted(idx0 + idx1)

        grid_train_x = x[grid_train_samples, :]
        grid_train_y = y[grid_train_samples, :]

        while grid_train_x.shape[0] < 60:
            grid_train_x = np.concatenate([grid_train_x, grid_train_x], axis=0)
            grid_train_y = np.concatenate([grid_train_y, grid_train_y], axis=0)

        while x.shape[0] < 60:
            x = np.concatenate([x, x], axis=0)
            y = np.concatenate([y, y], axis=0)

        grid_train_y = np.ravel(grid_train_y)

        response_rate = sum(y) / len(y)
        print('Response Rate', response_rate)

        if not self.is_trained:
            # Two-step cross-validation for hyperparameter selection
            print('-----------------Search Regularization Params---------------------')
            print(_)
            if response_rate < 0.005:
                depth_choice = [5]
            else:
                depth_choice = [8, 10]

            params = {
                'min_split_gain': [0.1],
                'max_depth': depth_choice,
                'min_child_weight': [5, 10, 30, 50, 60, 80, 100],
                'colsample_bytree': [0.6, 0.7],
                'learning_rate': [0.3],
                'subsample': [0.8],
                'num_leaves': [80],
                'objective': self.objective
            }

            cv_start = time.time()
            search_iter = 14
            n_estimators_choice = [50]
            _, best_param = search(self.lgbm,
                                   params,
                                   search_iter,
                                   n_estimators_choice,
                                   grid_train_x, grid_train_y)

            print('-----------------Search Learning Rate---------------------')
            print(_)
            for key, value in best_param.items():
                best_param[key] = [value]
            best_param['learning_rate'] = [0.03, 0.045, 0.06, 0.075, 0.85, 0.95, 0.105, 0.12]
            n_estimators_choice = [100, 150, 200]
            search_iter = 16

            self.clf, best_param = search(self.lgbm,
                                          best_param,
                                          search_iter,
                                          n_estimators_choice,
                                          grid_train_x, grid_train_y)

            print('self.clf', self.clf)
            cv_end = time.time()
            self.cv_time = cv_end - cv_start
            self.is_trained = True

        # Fit Model
        self.clf.fit(x, np.ravel(y))

        pre_model_name = []
        for file in os.listdir(os.getcwd()):
            if file.endswith("_lgb.txt"):
                pre_model_name.append(file)
        self.save_filename = str(len(pre_model_name) + 1) + '_lgb.txt'
        self.clf.booster_.save_model(self.save_filename)

        print("The whole available data is: ")
        print("Real-FIT: dim(X)= [{:d}, {:d}]".format(x.shape[0], x.shape[1]))

        print('Feature Importance:')
        print(self.clf.feature_importances_)

    def predict(self, x_test):
        """
        This function should provide predictions of labels on (test) data.
        The function predict eventually casdn return probabilities or continuous values.
        """
        booster = lgb.Booster(model_file=self.save_filename)
        y = booster.predict(x_test)
        return y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self


class TabularRegressor(TabularSupervised):
    """TabularRegressor class.

    It is used for tabular data regression with lightgbm regressor.
    """
    def __init__(self):
        super().__init__()
        self.objective = 'regression'
        self.lgbm = LGBMRegressor(silent=False,
                                  verbose=-1,
                                  n_jobs=1,
                                  objective=self.objective)


class TabularClassifier(TabularSupervised):
    """TabularClassifier class.

     It is used for tabular data classification with lightgbm classifier.
    """