import numpy as np
from autokeras import TabularClassifier

if __name__ == '__main__':
    ntime, nnum, ncat, nmvc = 4, 10, 8, 2
    nsample = 1000
    x_num = np.random.random([nsample, nnum])
    x_time = np.random.random([nsample, ntime])
    x_cat = np.random.randint(0, 10, [nsample, ncat])
    x_mvc = np.random.randint(0, 10, [nsample, nmvc])

    x_all = np.concatenate([x_num, x_time, x_cat, x_mvc], axis=1)
    x_train = x_all[:int(nsample * 0.8), :]
    x_test = x_all[int(nsample * 0.8):, :]

    y_all = np.random.randint(0, 2, nsample)
    y_train = y_all[:int(nsample * 0.8)]
    y_test = y_all[int(nsample * 0.8):]

    clf = TabularClassifier()
    datainfo = {'loaded_feat_types': [ntime, nnum, ncat, nmvc]}
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60, datainfo=datainfo)

    results = clf.evaluate(x_test, y_test)
    print(results)
