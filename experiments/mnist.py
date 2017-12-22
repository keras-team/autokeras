import sys
from keras.datasets import mnist
from sklearn.model_selection import StratifiedKFold
import numpy

from autokeras.classifier import ImageClassifier

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    clf = ImageClassifier(searcher_type=sys.argv[1])
    clf.fit(x_train, y_train)
    y = clf.evaluate(x_test, y_test)
    # MLP for Pima Indians Dataset with 10-fold cross validation

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load pima indians dataset
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, Y):
        # create model
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
        # evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
