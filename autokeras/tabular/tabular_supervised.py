import os
import re
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
import pickle
import numpy as np
from os.path import isfile
import random
import time

from autokeras.supervised import Supervised
from autokeras.tabular.tabular_preprocessor import TabularPreprocessor
from autokeras.utils import rand_temp_folder_generator


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
    if lgbm.objective == 'binary':
        score_metric = 'roc_auc'
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    elif lgbm.objective == 'multiclass':
        score_metric = 'f1_weighted'
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    elif lgbm.objective == 'regression':
        score_metric = 'neg_mean_squared_error'
        skf = KFold(n_splits=folds, shuffle=True, random_state=1001)

    random_search = RandomizedSearchCV(lgbm, param_distributions=params, n_iter=search_iter, scoring=score_metric,
                                       n_jobs=1, cv=skf, verbose=0, random_state=1001)

    random_search.fit(x, y)

    return random_search.best_estimator_, random_search.best_params_


class TabularSupervised(Supervised):
    def __init__(self, path=None):
        """
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        """
        super().__init__()
        self.is_trained = False
        self.clf = None
        self.save_filename = None
        self.objective = 'multiclass'
        self.tabular_preprocessor = None
        if path is None:
            path = rand_temp_folder_generator()
            print('Path:', path)

        self.path = path
        self.time_limit = None
        self.datainfo = None

    def fit(self, x, y, time_limit=None, datainfo=None):
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

        if time_limit is None:
            time_limit = 24 * 60 * 60
        if datainfo is None:
            datainfo = {'loaded_feat_types': [0] * 4}
        self.time_limit = time_limit
        self.datainfo = datainfo

        if self.objective == 'multiclass' or self.objective == 'binary':
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

        elif self.objective == 'regression':
            self.lgbm = LGBMRegressor(silent=False,
                                      verbose=-1,
                                      n_jobs=1,
                                      objective=self.objective)

        self.tabular_preprocessor = TabularPreprocessor()
        x = self.tabular_preprocessor.fit(x, y, self.time_limit, self.datainfo)

        if x.shape[1] == 0:
            raise ValueError("No feature exist!")

        if x.shape[0] > 6000:
            grid_train_perentage = 0.1
        else:
            grid_train_perentage = 1
        grid_N = int(x.shape[0] * grid_train_perentage)
        idx = random.sample(list(range(x.shape[0])), grid_N)

        grid_train_x = x[idx, :]
        grid_train_y = y[idx]

        while x.shape[0] < 60:
            x = np.concatenate([x, x], axis=0)
            y = np.concatenate([y, y], axis=0)

        response_rate = sum(y) / len(y)
        print('Response Rate', response_rate)

        if not self.is_trained:
            # Two-step cross-validation for hyperparameter selection
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

            cv_start = time.time()
            search_iter = 14
            n_estimators_choice = [50]
            _, best_param = search(self.lgbm,
                                   params,
                                   search_iter,
                                   n_estimators_choice,
                                   grid_train_x, grid_train_y)

            print('-----------------Search Learning Rate---------------------')
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
        self.clf.fit(x, y)

        pre_model_name = []
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        for file in os.listdir(self.path):
            if file.endswith("_lgb.txt"):
                pre_model_name.append(file)
        self.save_filename = self.path + '/' + str(len(pre_model_name) + 1) + '_lgb.txt'
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
        x_test = self.tabular_preprocessor.encode(x_test)
        if self.clf is not None:
            y = self.clf.predict(x_test)
        elif self.save_filename is not None:
            booster = lgb.Booster(model_file=self.save_filename)
            y = booster.predict(x_test)
        else:
            pre_model_name = []
            for file in os.listdir(self.path):
                if file.endswith("_lgb.txt"):
                    file_ind_pat = re.compile("(\d+)")
                    tmp_filename = int(file_ind_pat.findall(file)[0])
                    pre_model_name.append(tmp_filename)
            total_model = len(pre_model_name)
            if total_model == 0:
                raise ValueError("Tabular predictor does not exist")
            else:
                # Use the latest predictor
                self.save_filename = self.path + '/' + str(max(pre_model_name)) + '_lgb.txt'
                booster = lgb.Booster(model_file=self.save_filename)
                y = booster.predict(x_test)

        if y is None:
            raise ValueError("Tabular predictor does not exist")
        return y

    def evaluate(self, x_test, y_test):
        print('objective:', self.objective)
        y_pred = self.predict(x_test)
        if self.objective == 'binary':
            results = roc_auc_score(y_test, y_pred)
        elif self.objective == 'multiclass':
            results = f1_score(y_test, y_pred, average='weighted')
        elif self.objective == 'regression':
            results = mean_squared_error(y_test, y_pred)
        return results

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


class TabularClassifier(TabularSupervised):
    """TabularClassifier class.

     It is used for tabular data classification with lightgbm classifier.
    """
