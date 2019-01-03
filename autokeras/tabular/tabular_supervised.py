from abc import abstractmethod

import os
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
import numpy as np
import random

from autokeras.supervised import Supervised
from autokeras.tabular.tabular_preprocessor import TabularPreprocessor
from autokeras.utils import rand_temp_folder_generator, ensure_dir


class TabularSupervised(Supervised):
    def __init__(self, path=None, **kwargs):
        """
        Initialization function for tabular supervised learner.
        """
        super().__init__(**kwargs)
        self.is_trained = False
        self.clf = None
        self.objective = None
        self.tabular_preprocessor = None
        self.path = path if path is not None else rand_temp_folder_generator()
        ensure_dir(self.path)
        if self.verbose:
            print('Path:', path)
        self.save_filename = os.path.join(self.path, 'lgbm.txt')
        self.time_limit = None
        self.lgbm = None

    def search(self, search_space, search_iter, n_estimators, x, y):
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
        if self.verbose:
            print(params)
        folds = 3
        score_metric, skf = self.get_skf(folds)

        random_search = RandomizedSearchCV(self.lgbm, param_distributions=params, n_iter=search_iter,
                                           scoring=score_metric,
                                           n_jobs=1, cv=skf, verbose=0, random_state=1001)

        random_search.fit(x, y)
        self.clf = random_search.best_estimator_

        return random_search.best_params_

    @abstractmethod
    def get_skf(self, folds):
        pass

    def fit(self, x, y, time_limit=None, data_info=None):
        """
        This function should train the model parameters.

        Args:
            x: A numpy.ndarray instance containing the training data.
            y: training label vector.
            time_limit: remaining time budget.
            data_info: meta-features of the dataset, which is an numpy.ndarray describing the
             feature type of each column in raw_x. The feature type include:
                     'TIME' for temporal feature, 'NUM' for other numerical feature,
                     and 'CAT' for categorical feature.
        Both inputs X and y are numpy arrays.
        If fit is called multiple times on incremental data (train, test1, test2, etc.)
        you should warm-start your training from the pre-trained model. Past data will
        NOT be available for re-training.
        """

        if time_limit is None:
            time_limit = 24 * 60 * 60
        self.time_limit = time_limit

        self.init_lgbm(y)

        self.tabular_preprocessor = TabularPreprocessor()

        if x.shape[1] == 0:
            raise ValueError("No feature exist!")

        x = self.tabular_preprocessor.fit(x, y, self.time_limit, data_info)

        if x.shape[0] > 600:
            grid_train_percentage = max(600.0 / x.shape[0], 0.1)
        else:
            grid_train_percentage = 1
        grid_n = int(x.shape[0] * grid_train_percentage)
        idx = random.sample(list(range(x.shape[0])), grid_n)

        grid_train_x = x[idx, :]
        grid_train_y = y[idx]

        while x.shape[0] < 60:
            x = np.concatenate([x, x], axis=0)
            y = np.concatenate([y, y], axis=0)

        response_rate = sum(y) / len(y)

        if not self.is_trained:
            # Two-step cross-validation for hyperparameter selection
            if self.verbose:
                print('-----------------Search Regularization Params---------------------')
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
            }

            search_iter = 14
            n_estimators_choice = [50]
            best_param = self.search(
                params,
                search_iter,
                n_estimators_choice,
                grid_train_x, grid_train_y)

            if self.verbose:
                print('-----------------Search Learning Rate---------------------')
            for key, value in best_param.items():
                best_param[key] = [value]
            best_param['learning_rate'] = [0.03, 0.045, 0.06, 0.075, 0.85, 0.95, 0.105, 0.12]
            n_estimators_choice = [100, 150, 200]
            search_iter = 16

            self.search(
                best_param,
                search_iter,
                n_estimators_choice,
                grid_train_x, grid_train_y)

            if self.verbose:
                print('self.clf', self.clf)
            self.is_trained = True

        # Fit Model
        self.clf.fit(x, y)

        self.clf.booster_.save_model(self.save_filename)

        if self.verbose:
            print("The whole available data is: ")
            print("Real-FIT: dim(X)= [{:d}, {:d}]".format(x.shape[0], x.shape[1]))

            print('Feature Importance:')
            print(self.clf.feature_importances_)

    @abstractmethod
    def init_lgbm(self, y):
        pass
    def predict(self, x_test):
        """
        This function should provide predictions of labels on (test) data.
        The function predict eventually casdn return probabilities or continuous values.
        """
        x_test = self.tabular_preprocessor.encode(x_test)
        y = self.clf.predict(x_test)
        if y is None:
            raise ValueError("Tabular predictor does not exist")
        return y

    @abstractmethod
    def evaluate(self, x_test, y_test):
        pass

    def final_fit(self, x_train, y_train, x_test=None, y_test=None, trainer_args=None, retrain=False):
        x_train = self.tabular_preprocessor.encode(x_train)
        self.clf.fit(x_train, y_train)


class TabularRegressor(TabularSupervised):
    """TabularRegressor class.
    It is used for tabular data regression with lightgbm regressor.
    """
    def __init__(self, path=None):
        super().__init__(path)
        self.objective = 'regression'

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return mean_squared_error(y_test, y_pred)

    def init_lgbm(self, y):
        self.lgbm = LGBMRegressor(silent=False,
                                  verbose=-1,
                                  n_jobs=1,
                                  objective=self.objective)

    def get_skf(self, folds):
        return 'neg_mean_squared_error', KFold(n_splits=folds, shuffle=True, random_state=1001)


class TabularClassifier(TabularSupervised):
    """TabularClassifier class.
     It is used for tabular data classification with lightgbm classifier.
    """

    def init_lgbm(self, y):
        n_classes = len(set(y))
        if n_classes == 2:
            self.objective = 'binary'
            self.lgbm = LGBMClassifier(silent=False,
                                       verbose=-1,
                                       n_jobs=1,
                                       objective=self.objective)
        else:
            self.objective = 'multiclass'
            self.lgbm = LGBMClassifier(silent=False,
                                       verbose=-1,
                                       n_jobs=1,
                                       num_class=n_classes,
                                       objective=self.objective)

    def evaluate(self, x_test, y_test):
        if self.verbose:
            print('objective:', self.objective)
        y_pred = self.predict(x_test)
        results = None
        if self.objective == 'binary':
            results = roc_auc_score(y_test, y_pred)
        elif self.objective == 'multiclass':
            results = f1_score(y_test, y_pred, average='weighted')
        return results

    def get_skf(self, folds):
        if self.lgbm.objective == 'binary':
            score_metric = 'roc_auc'
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
        else:
            score_metric = 'f1_weighted'
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
        return score_metric, skf