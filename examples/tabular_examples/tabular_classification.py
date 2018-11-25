from autokeras import TabularClassifier
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    clf = TabularClassifier()
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60, datainfo=None)
    y_pred = clf.predict(x_test)
    AUC = roc_auc_score(y_test, y_pred)
    print(AUC)
