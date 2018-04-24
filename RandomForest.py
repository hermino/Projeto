# -*- coding: utf-8 -*-
import cv2
import glob
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

metascore = []

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
# emotions = ["happy", "surprise"]


def get_grid_search(clf, training_X, training_y, test_X, test_y):

    param_grid = {
        'n_estimators': [200, 500, 700],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [10, 20, 40, 50]
    }

    grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
    grid.fit(training_X, training_y)

    y_true, y_pred = test_y, grid.predict(test_X)

    bs = grid.best_score_
    bp = grid.best_params_
    cr = classification_report(y_true, y_pred)

    return bs, bp, cr

def get_files(emotion):
    files = glob.glob("base\dataset\\{}\\*".format(emotion))
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]
    test = files[-int(len(files) * 0.2):]
    return training, test


def make_sets():
    training_X = []
    training_y = []
    test_X = []
    test_y = []

    for emotion in emotions:

        training, test = get_files(emotion)

        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_X.append(gray)
            training_y.append(emotions.index(emotion))

        for item in test:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            test_X.append(gray)
            test_y.append(emotions.index(emotion))

    training_X = np.array(training_X)
    training_y = np.array(training_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    return training_X, training_y, test_X, test_y


def training_and_test(clf):
    training_X, training_y, test_X, test_y = make_sets()

    training_X = training_X.reshape(training_X.shape[0], 350*350)
    test_X = test_X.reshape(test_X.shape[0], 350*350)

    clf = clf.fit(training_X, training_y)
    predict = clf.predict(test_X)

    score = clf.score(test_X, test_y)
    cm = confusion_matrix(test_y, predict, labels=[0, 1, 2, 3, 4, 5, 6, 7])

    accuracy = accuracy_score(predict, test_y)

    bs, bp, cr = get_grid_search(clf, training_X, training_y, test_X, test_y)

    return accuracy, score, cm, bs, bp, cr


clf = RandomForestClassifier(bootstrap=False,
                             criterion='entropy',
                             max_depth=6,
                             max_features=51,
                             min_samples_leaf=5,
                             min_samples_split=16,
                             n_estimators=200
                             )

for i in range(0, 3):

    correct, score, cm, bs, bp, cr = training_and_test(clf)

    print('\033[1m Etapa: ' + '\033[0m{}\n'.format(i + 1))
    print ('\033[1m Score: ' + '\033[1;34m {} \n'.format(score) + '\033[0m')
    print('\033[1m Matriz de confusao:\n\n' + '\033[0m {} \n'.format(cm))
    print('\033[1m Best Score: ' + '\033[1;34m {}\n'.format(bs) + '\033[0m')
    print('\033[1m Best Params:\n\n' + '\033[0m {}\n'.format(bp))
    print('\033[1m Classification Report:\n\n' + '\033[0m {}'.format(cr))

    metascore.append(correct)

metascore = np.array(metascore)

print("\nAcuracy: %1.2f%%" % (metascore.mean()*100))
print("Margin of Error for More and Less: %1.2f%%" % (metascore.std()*100))
