import os
import numpy as np
from pandas import DataFrame
from scipy.stats import pearsonr
import multiprocessing as mp

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


def call_parallel(tasks):
    results = []
    for t in tasks:
        results.append(parallel_function(t[0], t[1], t[2]))
    return results


class TabularPreprocessor():
    def __init__(self):
        """
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        """
        self.num_cat_pair = {}

        self.total_samples = 0

        self.cat_to_int_label = {}
        self.n_first_batch_keys = {}
        self.high_level_cat_keys = []

        self.feature_add_high_cat = 0
        self.feature_add_cat_num = 0
        self.feature_add_cat_cat = 0
        self.order_num_cat_pair = {}

        self.rest = None
        self.budget = None
        self.datainfo = None

    def remove_useless(self, x):
        self.rest = np.where(np.max(x, 0) - np.min(x, 0) != 0)[0]
        return x[:, self.rest]

    def process_time(self, x):
        cols = range(self.ntime)
        if len(cols) > 10:
            cols = cols[:10]
        x_time = x[:, cols]
        for i in cols:
            for j in range(i+1, len(cols)):
                x = np.append(x, np.expand_dims(x_time[:, i]-x_time[:, j], 1), 1)
        return x

    def extract_data(self, F, ncat, nmvc):
        # only get numerical variables

        if type(F) == np.ndarray:
            ret = F
            n_rows = ret.shape[0]
            n_num_col = ret.shape[1] - ncat - nmvc
        else:
            n_rows = F['numerical'].shape[0]
            n_num_col = F['numerical'].shape[1]

            data_list = [F['numerical']]
            if ncat > 0:
                data_list.append(F['CAT'])
            if nmvc > 0:
                data_list.append(F['MV'])
            ret = np.concatenate(data_list, axis=1)

        n_cat_col = nmvc + ncat
        if n_cat_col <= 0:
            return ret.astype(np.float64)

        # preprocess (multi-value) categorical data
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
        if y is not None:
            mark = ntime + nnum

            for col_index in range(ntime + nnum, ntime + nnum + ncat + nmvc):
                if self.n_first_batch_keys[col_index] <= LEVEL_HIGH:
                    self.num_cat_pair[mark] = (col_index,)
                    mark += 1
                else:
                    self.num_cat_pair[mark] = (col_index, col_index)
                    mark += 1

            mark_1 = 0
            tasks_1 = []
            for i, cat_col_index1 in enumerate(self.high_level_cat_keys):
                for cat_col_index2 in self.high_level_cat_keys[i + 1:]:
                    tasks_1.append((X[:, (cat_col_index1, cat_col_index2)],
                                    [y, cat_col_index1, cat_col_index2, mark_1],
                                    'train_cat_cat'))
                    mark_1 += 1

            # pool = mp.Pool()
            # results = [pool.apply_async(parallel_function, t) for t in tasks_1]
            # all_results = [result.get() for result in results]
            # pool.close()
            # pool.join()
            all_results = call_parallel(tasks_1)

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
            print('num_cat_pair_1:', num_cat_pair_1)

            mark_2 = 0
            tasks_2 = []
            for cat_col_index in self.high_level_cat_keys:
                for num_col_index in range(ntime, ntime + nnum):
                    tasks_2.append((X[:, (num_col_index, cat_col_index)],
                                    [y, num_col_index, cat_col_index, mark_2],
                                    'train_num_cat'))
                    mark_2 += 1

            # pool = mp.Pool()
            # results = [pool.apply_async(parallel_function, t) for t in tasks_2]
            # all_results = [result.get() for result in results]
            # pool.close()
            # pool.join()
            all_results = call_parallel(tasks_2)

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
            print('num_cat_pair_2:', num_cat_pair_2)

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

        # pool = mp.Pool()
        # results = [pool.apply_async(parallel_function, t) for t in tasks]
        # all_num = X.shape[1] - ncat - nmvc
        # results = [X[:, :all_num]] + [result.get() for result in results]
        #
        # ret = np.concatenate(results, axis=1)
        # pool.close()
        # pool.join()

        results = call_parallel(tasks)
        all_num = X.shape[1] - ncat - nmvc
        results = [X[:, :all_num]] + results
        ret = np.concatenate(results, axis=1)

        return ret #, ret.shape[1] - all_num, 0

    def fit(self, F, y=None, time_limit=None, datainfo=None):
        """
        This function should train the model parameters.

        Args:
            x: A numpy.ndarray instance containing the training data.
            y: Training label matrix of dim num_train_samples * num_labels.
            datainfo: Meta-features of the dataset, which describe:
                     the number of four different features including:
                     time, numerical, categorical, and multi-value categorical.
        Both inputs X and y are numpy arrays.
        If fit is called multiple times on incremental data (train, test1, test2, etc.)
        you should warm-start your training from the pre-trained model. Past data will
        NOT be available for re-training.
        """
        # Get Meta-Feature
        self.budget = time_limit
        self.datainfo = datainfo
        [self.ntime, self.nnum, self.ncat, self.nmvc] = self.datainfo['loaded_feat_types']

        for col_index in range(self.nnum + self.ntime, self.nnum + self.ntime + self.ncat + self.nmvc):
            self.cat_to_int_label[col_index] = {}

        x = self.extract_data(F, self.ncat, self.nmvc)

        d_size = x.shape[0] * x.shape[1] / self.budget
        print('d_size', d_size)
        if d_size > 35000:
            self.feature_add_high_cat = 0
        else:
            self.feature_add_high_cat = 10

        for col_index in range(self.nnum + self.ntime, self.nnum + self.ntime + self.ncat + self.nmvc):
            self.n_first_batch_keys[col_index] = len(self.cat_to_int_label[col_index])
        high_level_cat_keys_tmp = sorted(self.n_first_batch_keys, key=self.n_first_batch_keys.get, reverse=True)[
                                  :self.feature_add_high_cat]
        for i in high_level_cat_keys_tmp:
            if self.n_first_batch_keys[i] > 1e2:
                self.high_level_cat_keys.append(i)

        print('hig_order_cat_pair:', self.high_level_cat_keys)
        print('n_first_batch_keys:', self.n_first_batch_keys)


        # Convert NaN to zeros
        x = np.nan_to_num(x)

        if datainfo is None:
            self.nnum = x.shape[1]

        # Encode high-order categorical data to numerical with frequency
        x = self.cat_to_num(x, self.ncat, self.nmvc, self.nnum, self.ntime, y)

        print('X.shape before remove_useless', x.shape)
        x = self.process_time(x)
        x = self.remove_useless(x)
        print('X.shape after remove_useless', x.shape)

        return x

    def encode(self, F, time_limit=None, datainfo=None):
        """
        This function should train the model parameters.

        Args:
            x: A numpy.ndarray instance containing the training data.
            y: Training label matrix of dim num_train_samples * num_labels.
            datainfo: Meta-features of the dataset, which describe:
                     the number of four different features including:
                     time, numerical, categorical, and multi-value categorical.
        Both inputs X and y are numpy arrays.
        If fit is called multiple times on incremental data (train, test1, test2, etc.)
        you should warm-start your training from the pre-trained model. Past data will
        NOT be available for re-training.
        """
        # Get Meta-Feature
        if time_limit is None:
            if self.budget is None:
                time_limit = 24 * 60 * 60
                self.budget = time_limit
        else:
            self.budget = time_limit

        if datainfo is None:
            if self.datainfo is None:
                datainfo = {'loaded_feat_types': [0] * 4}
                self.datainfo = datainfo
                [self.ntime, self.nnum, self.ncat, self.nmvc] = self.datainfo['loaded_feat_types']
        else:
            self.datainfo = datainfo


        x = self.extract_data(F, self.ncat, self.nmvc)

        # Convert NaN to zeros
        x = np.nan_to_num(x)
        if datainfo is None:
            self.nnum = x.shape[1]

        # Encode high-order categorical data to numerical with frequency
        x = self.cat_to_num(x, self.ncat, self.nmvc, self.nnum, self.ntime)

        print('X.shape before remove_useless', x.shape)
        x = self.process_time(x)
        if self.rest is not None:
            x = x[:, self.rest]
        print('X.shape after remove_useless', x.shape)

        return x