import autokeras as ak


def from_csv():
    clf = ak.StructuredDataClassifier(seed=5, max_trials=3)
    clf.fit(x='tests/fixtures/titanic/train.csv', y='survived', validation_split=0.2)
    clf.evaluate(x='tests/fixtures/titanic/eval.csv', y='survived')


if __name__ == "__main__":
    from_csv()
