import numpy as np
from autokeras import TabularClassifier

if __name__ == '__main__':
    nnum, ntime, ncat, nmvc = 10, 4, 8, 2
    nsample = 10000
    x_num = np.random.random([nsample, nnum])
    x_time = np.random.random([nsample, ntime])
    x_cat = np.random.randint(0, 10, [nsample, ncat])
    x_mvc = np.random.randint(0, 10, [nsample, nmvc])

    x_all = np.concatenate([x_num, x_time, x_cat, x_mvc], axis=1)
    x_train = x_all[:int(nsample*0.8), :]
    x_test = x_all[int(nsample * 0.8):, :]

    y = np.random.randint(0, 2, [nsample, 1])
    y_train = y[:int(nsample*0.8), :]
    y_test = x_all[int(nsample * 0.8):, :]

    clf = TabularClassifier()
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60, datainfo=None)
    AUC = clf.evaluate(x_test, y_test)
    print(AUC)
