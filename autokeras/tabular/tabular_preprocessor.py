import os
import numpy as np
from pandas import DataFrame
from scipy.stats import pearsonr
import multiprocessing as mp

LEVEL_HIGH = 32

class feature_model:
    def __init__(self, X):
        self.X = X

    def remove_useless(self):
        rest = np.where(np.max(self.X, 0) - np.min(self.X, 0) != 0)[0]
        self.X = self.X[:, rest]
        pre_model_name = []
        for file in os.listdir(os.getcwd()):
            if file.endswith("_lgb.npy"):
                pre_model_name.append(file)
        newname = str(len(pre_model_name) + 1) + '_lgb'
        np.save(newname, rest)

    def time(self, cols):
        if len(cols) > 10:
            cols = cols[:10]
        X_time = self.X[:, cols]
        for i in cols:
            for j in range(i+1, len(cols)):
                self.X = np.append(self.X, np.expand_dims(X_time[:, i]-X_time[:, j], 1), 1)



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


class tabular_preprocess():
    def __init__(self, datainfo):
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

        self.total_samples = 0
        self.is_trained = False

        self.cat_to_int_label = {}
        self.n_first_batch_keys = {}
        self.high_level_cat_keys = []

        self.lag = 0
        self.max_lag = 0
        self.save_predict = None

        self.batch_calculate = -1
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


    def extract_data(self, F, ncat, nmvc):
        # only get numerical variables
        n_rows = F['numerical'].shape[0]
        n_num_col = F['numerical'].shape[1]

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

            pool = mp.Pool()
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

            pool = mp.Pool()
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

        pool = mp.Pool()
        results = [pool.apply_async(parallel_function, t) for t in tasks]
        results = [X[:, :ntime + nnum]] + [result.get() for result in results]

        ret = np.concatenate(results, axis=1)
        pool.close()
        pool.join()
        return ret, ret.shape[1] - ntime - nnum, 0

    def fit(self, F, y, datainfo):
        """
        This function should train the model parameters.

        Args:
            x: A numpy.ndarray instance containing the training data.
            y: Training label matrix of dim num_train_samples * num_labels.
            datainfo: Meta-features of the dataset, which describe:
                     (i) the number of four different features including:
                        time, numerical, categorical, and multi-value categorical.
                     (ii) time budget.
        Both inputs X and y are numpy arrays.
        If fit is called multiple times on incremental data (train, test1, test2, etc.)
        you should warm-start your training from the pre-trained model. Past data will
        NOT be available for re-training.
        """

        # Mark the number of batches
        self.batch_calculate += 1

        # Get Meta-Feature
        [ntime, nnum, ncat, nmvc] = datainfo['loaded_feat_types']
        budget = datainfo['budget']

        if not self.is_trained:
            for col_index in range(nnum + ntime, nnum + ntime + ncat + nmvc):
                self.cat_to_int_label[col_index] = {}

        if type(F) == np.ndarray:
            X = F
        else:
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
        else:
            self.feature_add_cat_num = 10
            self.feature_add_cat_cat = 10

        # Convert NaN to zeros
        X = np.nan_to_num(X)

        # Encode high-order categorical data to numerical with frequency
        X, ncat, nmvc = self.cat_to_num(X, ncat, nmvc, nnum, ntime, np.ravel(y))

        print('X.shape before remove_useless', X.shape)
        feature = feature_model(X)
        feature.time(range(ntime))
        feature.remove_useless()
        X = feature.X
        print('X.shape after remove_useless', X.shape)

        return X
