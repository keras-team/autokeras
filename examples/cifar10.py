from keras.datasets import cifar10

from autokeras import ImageClassifier

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    clf = ImageClassifier(verbose=True, augment=True)
    clf.fit(x_train, y_train, time_limit=12*60*60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True,
                  trainer_args={'max_iter_num': 630, 'max_no_improvement_num': 630})
    y = clf.evaluate(x_test, y_test)

    print(y * 100)
