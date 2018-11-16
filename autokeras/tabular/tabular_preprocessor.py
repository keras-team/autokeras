import os
import numpy as np

class feature_model:
    def __init__(self, X):
        self.X = X

    def remove_useless(self):
        rest = np.where(np.max(self.X, 0) - np.min(self.X, 0) != 0)[0]
        self.X = self.X[:, rest]
        print(self.X.shape)
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
        print(self.X[:, cols].min(axis=0))
        print(self.X[:, cols].max(axis=0))
        print(self.X[:, cols].mean(axis=0))
