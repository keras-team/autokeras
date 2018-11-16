import os
import lightgbm as lgb
from lightgbm import LGBMRegressor
import multiprocessing as mp
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pickle
import numpy as np  # We recommend to use numpy arrays
from os.path import isfile
import random
import time
from pandas import DataFrame
from scipy.stats import pearsonr

from autokeras.tabular.tabular_preprocessor import feature_model
from autokeras.supervised import Supervised

ITERATION = 0
LEVEL_HIGH = 32


def parallel_function(labels, first_batch_keys, task):
    if task == 'label':
        if min(labels) > first_batch_keys:
            labels = labels - np.min(labels)
        return labels.reshape(labels.shape[0], 1)

    elif task == 'frequency':
        cat_dict = {}
        n_rows = labels.shape[0]
        labels = np.expand_dims(labels, axis=1)

        if min(labels) > first_batch_keys:
            labels = labels - np.min(labels)

        frequencies = np.zeros((n_rows, 1))

        for row_index in range(n_rows):
            key = labels[row_index, 0]
            if key in cat_dict:
                cat_dict[key] += 1
            else:
                cat_dict[key] = 1

        n_level = len(cat_dict)
        key_to_frequency = {}

        for key in cat_dict.keys():
            key_to_frequency[key] = cat_dict[key] / n_rows * n_level

        for row_index in range(n_rows):
            key = labels[row_index, 0]
            frequencies[row_index][0] = key_to_frequency[key]

        return frequencies
    elif task == 'num_cat':
        df = DataFrame(data=labels)
        return df.join(df.groupby(1)[0].mean(),
                       rsuffix='r',
                       on=1).values[:, -1:]
    elif task == 'cat_cat':
        df = DataFrame(data=labels)
        df[3] = list(range(len(labels)))
        return df.join(df.groupby([0, 1]).count(),
                       rsuffix='r',
                       on=(0, 1)).values[:, -1:]
    elif task == 'train_num_cat':
        y = first_batch_keys[0]
        df = DataFrame(data=labels)
        fe = df.join(df.groupby(1)[0].mean(),
                     rsuffix='r',
                     on=1).values[:, -1:]
        mu = abs(pearsonr(np.squeeze(np.array(fe)), y)[0])
        if np.isnan(mu):
            mu = 0
        return [[first_batch_keys[1], first_batch_keys[2], mu, mu], first_batch_keys[3]]

    elif task == 'train_cat_cat':
        y = first_batch_keys[0]
        df = DataFrame(data=labels)
        df[3] = list(range(len(labels)))
        fe = df.join(df.groupby([0, 1]).count(),
                     rsuffix='r',
                     on=(0, 1)).values[:, -1:]
        mu = abs(pearsonr(np.squeeze(np.array(fe)), y)[0])
        if np.isnan(mu):
            mu = 0
        return [[first_batch_keys[1], first_batch_keys[2], mu], first_batch_keys[3]]
    return None


def search(search_space, search_iter, n_estimators, x, y):
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
    # pdb.set_trace()
    params.update(search_space)
    print(params)
    xgb = LGBMRegressor(silent=False,
                        verbose=-1,
                        n_jobs=4,
                        # max_depth=6,
                        # n_estimators=500,
                        # tree_method='hist',
                        objective='binary')
    folds = 3
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    # skf = KFold(n_splits=folds, shuffle=True, random_state=1001)
    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=search_iter, scoring='roc_auc',
                                       n_jobs=1, cv=skf, verbose=0, random_state=1001)

    random_search.fit(x, y)

    return random_search.best_estimator_, random_search.best_params_


class Model(Supervised):
    def __init__(self, datainfo, timeinfo):
        """
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        """
        # Just print some info from the datainfo variable
        self.num_cat_pair = {}
        print("The Budget for this data set is: %d seconds" % datainfo['time_budget'])

        print("Loaded %d time features, "
              "%d numerical Features, "
              "%d categorical features and %d multi valued categorical variables" % (
                datainfo['loaded_feat_types'][0], datainfo['loaded_feat_types'][1], datainfo['loaded_feat_types'][2],
                datainfo['loaded_feat_types'][3]))
        overall_spenttime = time.time() - timeinfo[0]
        dataset_spenttime = time.time() - timeinfo[1]
        print("[***] Overall time spent %5.2f sec" % overall_spenttime)
        print("[***] Dataset time spent %5.2f sec" % dataset_spenttime)
        self.total_samples = 0
        self.is_trained = False
        # Here you may have parameters and hyper-parameters
        self.cat_to_int_label = {}
        self.n_first_batch_keys = {}
        self.high_level_cat_keys = []

        self.lag = 0
        self.max_lag = 0
        self.save_predict = None

        self.batch_calculate = -1
        self.former_batch_time = 0
        self.lag_results = []
        self.former_lag_results = []
        self.feature_add_high_cat = 0
        self.feature_add_cat_num = 0
        self.feature_add_cat_cat = 0
        self.order_num_cat_pair = None

        self.lag_bound = 10
        self.clf = None

        # Remove Previously Generated lgb results
        for file in os.listdir(os.getcwd()):
            if file.endswith("_lgb.txt") or file.endswith("_lgb.npy"):
                os.remove(file)

    def extract_data(self, F, ncat, nmvc, use_mark):
        # only get numerical variables
        n_rows = F['numerical'].shape[0]
        n_num_col = F['numerical'].shape[1]
        if use_mark:
            return np.concatenate((F['numerical'], self.engineered_data), axis=1).astype(np.float64)

        data_list = [F['numerical']]
        if ncat > 0:
            data_list.append(F['CAT'].values)
        if nmvc > 0:
            data_list.append(F['MV'].values)
        ret = np.concatenate(data_list, axis=1)

        n_cat_col = nmvc + ncat
        if n_cat_col <= 0:
            return ret.astype(np.float64)

        # preprocess multi-value categorical data
        for col_index in range(n_num_col, n_num_col + n_cat_col):
            for row_index in range(n_rows):
                key = str(ret[row_index, col_index])
                if key in self.cat_to_int_label[col_index]:
                    ret[row_index, col_index] = self.cat_to_int_label[col_index][key]
                    continue
                new_value = len(self.cat_to_int_label[col_index])
                self.cat_to_int_label[col_index][key] = new_value
                ret[row_index, col_index] = new_value

        return ret.astype(np.float64)

    def cat_to_num(self, X, ncat, nmvc, nnum, ntime, y=None):
        if not self.is_trained:
            mark = ntime + nnum

            for col_index in range(ntime + nnum, ntime + nnum + ncat + nmvc):
                if self.n_first_batch_keys[col_index] <= LEVEL_HIGH:
                    self.num_cat_pair[mark] = (col_index,)
                    print(self.num_cat_pair[mark])
                    mark += 1
                else:
                    self.num_cat_pair[mark] = (col_index, col_index)
                    print(self.num_cat_pair[mark])
                    mark += 1

            mark_1 = 0
            tasks_1 = []
            for i, cat_col_index1 in enumerate(self.high_level_cat_keys):
                for cat_col_index2 in self.high_level_cat_keys[i + 1:]:
                    tasks_1.append((X[:, (cat_col_index1, cat_col_index2)],
                                    [y, cat_col_index1, cat_col_index2, mark_1],
                                    'train_cat_cat'))
                    mark_1 += 1

            pool = mp.Pool(processes=4)
            results = [pool.apply_async(parallel_function, t) for t in tasks_1]
            all_results = [result.get() for result in results]
            pool.close()
            pool.join()

            num_cat_pair_1 = {}
            pearsonr_dict_1 = {}
            for result in all_results:
                if result[0][-1] > 0.001:
                    pearsonr_dict_1[result[1]] = result[0][-1]
                    num_cat_pair_1[result[1]] = result[0]
            pearsonr_high_1 = sorted(pearsonr_dict_1, key=pearsonr_dict_1.get, reverse=True)[:self.feature_add_cat_cat]
            num_cat_pair_1 = {key: num_cat_pair_1[key] for key in pearsonr_high_1}
            num_cat_pair_1 = {i + mark: num_cat_pair_1[key] for i, key in enumerate(num_cat_pair_1)}
            self.num_cat_pair.update(num_cat_pair_1)
            mark += len(pearsonr_high_1)

            print('num_cat_pair_1: ', num_cat_pair_1)

            mark_2 = 0
            tasks_2 = []
            for cat_col_index in self.high_level_cat_keys:
                for num_col_index in range(ntime, ntime + nnum):
                    tasks_2.append((X[:, (num_col_index, cat_col_index)],
                                    [y, num_col_index, cat_col_index, mark_2],
                                    'train_num_cat'))
                    mark_2 += 1

            pool = mp.Pool(processes=4)
            results = [pool.apply_async(parallel_function, t) for t in tasks_2]
            all_results = [result.get() for result in results]
            pool.close()
            pool.join()

            num_cat_pair_2 = {}
            pearsonr_dict_2 = {}
            for result in all_results:
                if result[0][-1] > 0.001:
                    pearsonr_dict_2[result[1]] = result[0][-1]
                    num_cat_pair_2[result[1]] = result[0]
            pearsonr_high_2 = sorted(pearsonr_dict_2, key=pearsonr_dict_2.get, reverse=True)[:self.feature_add_cat_num]
            num_cat_pair_2 = {key: num_cat_pair_2[key] for key in pearsonr_high_2}
            num_cat_pair_2 = {i + mark: num_cat_pair_2[key] for i, key in enumerate(num_cat_pair_2)}
            self.num_cat_pair.update(num_cat_pair_2)
            self.order_num_cat_pair = sorted(list(self.num_cat_pair.keys()))
            print(self.num_cat_pair)

            print('num_cat_pair_2: ', num_cat_pair_2)

        tasks = []
        for key in self.order_num_cat_pair:
            if len(self.num_cat_pair[key]) == 1:
                (col_index,) = self.num_cat_pair[key]
                tasks.append((X[:, col_index], self.n_first_batch_keys[col_index], 'label'))
            if len(self.num_cat_pair[key]) == 2:
                (col_index, col_index) = self.num_cat_pair[key]
                tasks.append((X[:, col_index], self.n_first_batch_keys[col_index], 'frequency'))
            if len(self.num_cat_pair[key]) == 3:
                (cat_col_index1, cat_col_index2, mu) = self.num_cat_pair[key]
                tasks.append((X[:, (cat_col_index1,
                                    cat_col_index2)], self.n_first_batch_keys[cat_col_index1], 'cat_cat'))
            elif len(self.num_cat_pair[key]) == 4:
                (num_col_index, cat_col_index, mu, a) = self.num_cat_pair[key]
                tasks.append((X[:, (num_col_index, cat_col_index)], self.n_first_batch_keys[cat_col_index], 'num_cat'))

        pool = mp.Pool(processes=4)
        results = [pool.apply_async(parallel_function, t) for t in tasks]
        results = [X[:, :ntime + nnum]] + [result.get() for result in results]

        ret = np.concatenate(results, axis=1)
        pool.close()
        pool.join()
        return ret, ret.shape[1] - ntime - nnum, 0

    def fit(self, F, y, datainfo, timeinfo):
        """
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        If fit is called multiple times on incremental data (train, test1, test2, etc.)
        you should warm-start your training from the pre-trained model. Past data will
        NOT be available for re-training.
        """

        # Mark the number of batches
        self.batch_calculate += 1

        # Get Meta-Feature
        [ntime, nnum, ncat, nmvc] = datainfo['loaded_feat_types']

        budget = datainfo['time_budget'] - 50

        if not self.is_trained:
            for col_index in range(nnum + ntime, nnum + ntime + ncat + nmvc):
                self.cat_to_int_label[col_index] = {}
        X = self.extract_data(F, ncat, nmvc, self.is_trained)

        d_size = X.shape[0] * X.shape[1] / budget
        if d_size > 35000:
            self.feature_add_high_cat = 0
        else:
            self.feature_add_high_cat = 10

        if not self.is_trained:
            for col_index in range(nnum + ntime, nnum + ntime + ncat + nmvc):
                self.n_first_batch_keys[col_index] = len(self.cat_to_int_label[col_index])
            high_level_cat_keys_tmp = sorted(self.n_first_batch_keys, key=self.n_first_batch_keys.get, reverse=True)[
                                      :self.feature_add_high_cat]
            for i in high_level_cat_keys_tmp:
                if self.n_first_batch_keys[i] > 1e2:
                    self.high_level_cat_keys.append(i)

        print('d_size', d_size)
        if d_size > 35000:
            self.feature_add_cat_num = 0
            self.feature_add_cat_cat = 0
        elif d_size > 23000:
            self.feature_add_cat_num = 10
            self.feature_add_cat_cat = 10
        elif d_size > 15000:
            self.feature_add_cat_num = 15
            self.feature_add_cat_cat = 15
        else:
            self.feature_add_cat_num = 50
            self.feature_add_cat_cat = 50

        overall_spenttime = time.time() - timeinfo[0]
        dataset_spenttime = time.time() - timeinfo[1]

        print("[***] Overall time spent %5.2f sec" % overall_spenttime)
        print("[***] Dataset time spent %5.2f sec" % dataset_spenttime)

        ###################
        # Time Controller #
        ###################
        if self.is_trained:
            print(budget - dataset_spenttime)
            print(9 - self.batch_calculate)
            tt = dataset_spenttime - self.former_batch_time
            if self.batch_calculate == 1:
                tt -= self.cv_time
                tt -= self.FE_time
            print(tt)
            print((9 - self.batch_calculate) * tt)
            if (budget - dataset_spenttime) < tt * (9 - self.batch_calculate):
                print('No!')
                self.max_lag -= 1
                if self.max_lag < 1:
                    print('No!!!')
                    self.max_lag = 1
            else:
                print('Yes')
                self.max_lag += 1

        if self.lag > self.max_lag:
            self.lag = self.max_lag

        self.former_batch_time = dataset_spenttime

        ############################
        # Lag and Weight Selection #
        ############################
        if self.is_trained:
            if self.batch_calculate > 1:
                print('QQ')
                print(self.lag_results)
                self.former_lag_results = np.array([self.lag_results[0] * 1.1] + list(self.lag_results))
            self.lag_results = []
            for i in range(self.lag):
                self.lag_results.append(roc_auc_score(y, self.save_predict[:, i]))
            self.lag_results = np.array(self.lag_results)
            print(self.lag_results)
            self.lag_results = 2 * self.lag_results - 1
            print(self.lag_results)

        if self.batch_calculate > 1:
            self.former_lag_results = self.former_lag_results[0:self.save_predict.shape[1]]
            cum_results = np.cumsum(self.former_lag_results * self.save_predict, axis=1)
            tmp_auc = []
            for i in range(cum_results.shape[1]):
                tmp_auc.append(roc_auc_score(y, cum_results[:, i]))
            print(np.array(tmp_auc) * 2 - 1)
            tmp_lag = tmp_auc.index(max(tmp_auc))
            if tmp_lag < self.lag - 1:
                self.lag = tmp_lag
            if self.lag < 1:
                self.lag = 1
            print('Current Lag!')
            print(self.lag)
            print('Tmp Lag!')
            print(tmp_lag)

        # convert NaN to zeros
        X = np.nan_to_num(X)

        start = time.time()
        if not self.is_trained:
            X, ncat, nmvc = self.cat_to_num(X, ncat, nmvc, nnum, ntime, np.ravel(y))
        end = time.time()
        self.FE_time = end - start
        print('Feature Engineering time: {}'.format(self.FE_time))

        print('X.shape before remove_useless', X.shape)

        feature = feature_model(X)
        feature.time(range(ntime))
        feature.remove_useless()
        X = feature.X

        print('X.shape after remove_useless', X.shape)

        num_train_samples = y.shape[0]

        overall_spenttime = time.time() - timeinfo[0]
        dataset_spenttime = time.time() - timeinfo[1]

        y0 = np.where(y == 0)[0]
        y1 = np.where(y == 1)[0]
        N0 = len(y0)
        N1 = len(y1)

        if X.shape[0] > 600:
            grid_train_perentage = 0.1
        else:
            grid_train_perentage = 1
        grid_N = int(X.shape[0] * grid_train_perentage)
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

        grid_train_X = X[grid_train_samples, :]
        grid_train_y = y[grid_train_samples, :]

        while grid_train_X.shape[0] < 60:
            grid_train_X = np.concatenate([grid_train_X, grid_train_X], axis=0)
            grid_train_y = np.concatenate([grid_train_y, grid_train_y], axis=0)

        while X.shape[0] < 60:
            X = np.concatenate([X, X], axis=0)
            y = np.concatenate([y, y], axis=0)

        grid_train_y = np.ravel(grid_train_y)

        response_rate = sum(y) / len(y)
        print('Response Rate', response_rate)

        # subsample the data for efficient processing
        if not self.is_trained:

            # if sum(y) / len(y) < 0.005:
            #     self.lag_bound = 2

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
                # 'n_estimators': [100],
                'subsample': [0.8],
                'num_leaves': [80],
            }

            cv_start = time.time()
            search_iter = 14
            n_estimators_choice = [50]
            _, best_param = search(params,
                                   search_iter,
                                   n_estimators_choice,
                                   grid_train_X, grid_train_y)
            print('-----------------Search Regularization Params---------------------')
            print(_)
            for key, value in best_param.items():
                best_param[key] = [value]

            print('-----------------Search Learning Rate---------------------')
            print(_)
            best_param['learning_rate'] = [0.03, 0.045, 0.06, 0.075, 0.85, 0.95, 0.105, 0.12]
            n_estimators_choice = [100, 150, 200]
            search_iter = 16

            self.clf, best_param = search(best_param,
                                          search_iter,
                                          n_estimators_choice,
                                          grid_train_X, grid_train_y)

            print('self.clf', self.clf)

            cv_end = time.time()
            self.cv_time = cv_end - cv_start

            self.is_trained = True

        self.DataX = X
        self.DataY = y

        # Fit Model
        self.clf.fit(self.DataX, np.ravel(self.DataY))

        pre_model_name = []
        for file in os.listdir(os.getcwd()):
            if file.endswith("_lgb.txt"):
                pre_model_name.append(file)
        newname = str(len(pre_model_name) + 1) + '_lgb.txt'
        self.clf.booster_.save_model(newname)

        print("The whole available data is: ")
        print("Real-FIT: dim(X)= [{:d}, {:d}]".format(self.DataX.shape[0], self.DataX.shape[1]))

        print('Feature Importance:')
        print(self.clf.feature_importances_)

    def predict(self, F, datainfo, timeinfo):
        """
        This function should provide predictions of labels on (test) data.
        Here we just return random values...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves.
        The function predict eventually casdn return probabilities or continuous values.
        """

        # Get Meta-Feature
        [ntime, nnum, ncat, nmvc] = datainfo['loaded_feat_types']
        budget = datainfo['time_budget']

        X = self.extract_data(F, ncat, nmvc, False)

        overall_spenttime = time.time() - timeinfo[0]
        dataset_spenttime = time.time() - timeinfo[1]

        print("[***] Overall time spent %5.2f sec" % overall_spenttime)
        print("[***] Dataset time spent %5.2f sec" % dataset_spenttime)

        # convert NaN to zeros
        X = np.nan_to_num(X)

        num_test_samples = X.shape[0]

        X, ncat, nmvc = self.cat_to_num(X, ncat, nmvc, nnum, ntime)
        subtt_cols = range(ntime + nnum, ntime + nnum + ncat + nmvc)
        self.engineered_data = X[:, subtt_cols]

        feature = feature_model(X)
        feature.time(range(ntime))
        X = feature.X

        y = 0
        pre_model_name = []
        for file in os.listdir(os.getcwd()):
            if file.endswith("_lgb.txt"):
                pre_model_name.append(file)

        total_model = len(pre_model_name)

        self.save_predict = np.zeros([num_test_samples, self.lag + 1])
        for name in pre_model_name:

            if int(name[0]) < total_model - self.lag:
                continue

            booster = lgb.Booster(model_file=name)
            selected_feature_file = name[0] + '_lgb.npy'
            tmp_y = booster.predict(
                X[:, np.load(selected_feature_file)]
            )
            y += tmp_y
            print(name)
            self.save_predict[:, total_model - int(name[0])] = tmp_y

        if total_model > 1:
            y = self.save_predict[:, 0] * np.max(self.lag_results[0]) * 1.1

            for i in range(1, self.save_predict.shape[1]):
                y += self.lag_results[i - 1] * self.save_predict[:, i]

        self.lag += 1
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
