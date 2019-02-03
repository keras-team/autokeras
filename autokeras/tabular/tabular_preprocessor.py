import numpy as np
from pandas import DataFrame
from scipy.stats import pearsonr

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


class TabularPreprocessor:
    def __init__(self):
        """
        Initialization function for tabular preprocessor.
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
        self.data_info = None
        self.n_time = None
        self.n_num = None
        self.n_cat = None

    def remove_useless(self, x):
        self.rest = np.where(np.max(x, 0) - np.min(x, 0) != 0)[0]
        return x[:, self.rest]

    def process_time(self, x):
        cols = range(self.n_time)
        if len(cols) > 10:
            cols = cols[:10]
        x_time = x[:, cols]
        for i in cols:
            for j in range(i + 1, len(cols)):
                x = np.append(x, np.expand_dims(x_time[:, i] - x_time[:, j], 1), 1)
        return x

    def extract_data(self, raw_x):
        # only get numerical variables
        ret = np.concatenate([raw_x['TIME'], raw_x['NUM'], raw_x['CAT']], axis=1)
        n_rows = ret.shape[0]
        n_num_col = ret.shape[1] - self.n_cat

        n_cat_col = self.n_cat
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

    def cat_to_num(self, x, y=None):
        if y is not None:
            mark = self.n_time + self.n_num

            for col_index in range(self.n_time + self.n_num, self.n_time + self.n_num + self.n_cat):
                if self.n_first_batch_keys[col_index] <= LEVEL_HIGH:
                    self.num_cat_pair[mark] = (col_index,)
                    mark += 1
                else:
                    self.num_cat_pair[mark] = (col_index, col_index)
                    mark += 1

            mark_1 = 0
            tasks = []
            for i, cat_col_index1 in enumerate(self.high_level_cat_keys):
                for cat_col_index2 in self.high_level_cat_keys[i + 1:]:
                    tasks.append((x[:, (cat_col_index1, cat_col_index2)],
                                  [y, cat_col_index1, cat_col_index2, mark_1],
                                  'train_cat_cat'))
                    mark_1 += 1

            all_results = call_parallel(tasks)

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

            mark_2 = 0
            tasks_2 = []
            for cat_col_index in self.high_level_cat_keys:
                for num_col_index in range(self.n_time, self.n_time + self.n_num):
                    tasks_2.append((x[:, (num_col_index, cat_col_index)],
                                    [y, num_col_index, cat_col_index, mark_2],
                                    'train_num_cat'))
                    mark_2 += 1

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
                tasks.append((x[:, col_index], self.n_first_batch_keys[col_index], 'label'))
            if len(self.num_cat_pair[key]) == 2:
                (col_index, col_index) = self.num_cat_pair[key]
                tasks.append((x[:, col_index], self.n_first_batch_keys[col_index], 'frequency'))
            if len(self.num_cat_pair[key]) == 3:
                (cat_col_index1, cat_col_index2, mu) = self.num_cat_pair[key]
                tasks.append((x[:, (cat_col_index1,
                                    cat_col_index2)], self.n_first_batch_keys[cat_col_index1], 'cat_cat'))
            elif len(self.num_cat_pair[key]) == 4:
                (num_col_index, cat_col_index, mu, a) = self.num_cat_pair[key]
                tasks.append((x[:, (num_col_index, cat_col_index)], self.n_first_batch_keys[cat_col_index], 'num_cat'))

        results = call_parallel(tasks)
        all_num = x.shape[1] - self.n_cat
        results = [x[:, :all_num]] + results
        ret = np.concatenate(results, axis=1)

        return ret

    def fit(self, raw_x, y, time_limit, data_info):
        """
        This function should train the model parameters.

        Args:
            raw_x: a numpy.ndarray instance containing the training data.
            y: training label vector.
            time_limit: remaining time budget.
            data_info: meta-features of the dataset, which is an numpy.ndarray describing the
             feature type of each column in raw_x. The feature type include:
                     'TIME' for temporal feature, 'NUM' for other numerical feature,
                     and 'CAT' for categorical feature.
        """
        # Get Meta-Feature
        self.budget = time_limit
        self.data_info = data_info if data_info is not None else self.extract_data_info(raw_x)
        print('QQ: {}'.format(self.data_info))

        self.n_time = sum(self.data_info == 'TIME')
        self.n_num = sum(self.data_info == 'NUM')
        self.n_cat = sum(self.data_info == 'CAT')

        self.total_samples = raw_x.shape[0]

        print('QQ1: {}'.format(self.n_time))
        print('QQ2: {}'.format(self.n_num))
        print('QQ3: {}'.format(self.n_cat))
        raw_x = {'TIME': raw_x[:, self.data_info == 'TIME'],
                 'NUM': raw_x[:, self.data_info == 'NUM'],
                 'CAT': raw_x[:, self.data_info == 'CAT']}


        for col_index in range(self.n_num + self.n_time, self.n_num + self.n_time + self.n_cat):
            self.cat_to_int_label[col_index] = {}

        x = self.extract_data(raw_x)

        d_size = x.shape[0] * x.shape[1] / self.budget
        if d_size > 35000:
            self.feature_add_high_cat = 0
        else:
            self.feature_add_high_cat = 10

        # Iterate cat features
        for col_index in range(self.n_num + self.n_time, self.n_num + self.n_time + self.n_cat):
            self.n_first_batch_keys[col_index] = len(self.cat_to_int_label[col_index])
        high_level_cat_keys_tmp = sorted(self.n_first_batch_keys, key=self.n_first_batch_keys.get, reverse=True)[
                                  :self.feature_add_high_cat]
        for i in high_level_cat_keys_tmp:
            if self.n_first_batch_keys[i] > 1e2:
                self.high_level_cat_keys.append(i)

        # Convert NaN to zeros
        x = np.nan_to_num(x)

        # Encode high-order categorical data to numerical with frequency
        x = self.cat_to_num(x, y)

        x = self.process_time(x)
        x = self.remove_useless(x)

        return x

    def encode(self, raw_x, time_limit=None):
        """
        This function should train the model parameters.

        Args:
            raw_x: a numpy.ndarray instance containing the training/testing data.
            time_limit: remaining time budget.
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

        raw_x = {'TIME': raw_x[:, self.data_info == 'TIME'],
                 'NUM': raw_x[:, self.data_info == 'NUM'],
                 'CAT': raw_x[:, self.data_info == 'CAT']}
        x = self.extract_data(raw_x)

        # Convert NaN to zeros
        x = np.nan_to_num(x)

        # Encode high-order categorical data to numerical with frequency
        x = self.cat_to_num(x)

        x = self.process_time(x)
        if self.rest is not None:
            x = x[:, self.rest]
        return x

    @staticmethod
    def extract_data_info(raw_x):
        """
        This function extracts the data info automatically based on the type of each feature in raw_x.

        Args:
            raw_x: a numpy.ndarray instance containing the training data.
        """
        data_info = []
        row_num, col_num = raw_x.shape
        for col_idx in range(col_num):
            try:
                raw_x[:, col_idx].astype(np.float)
                data_info.append('NUM')
            except:
                data_info.append('CAT')
        return np.array(data_info)
