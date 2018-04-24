# -*- coding: utf-8 -*-
import cv2
import glob
import random
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

metascore = []

# emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
emotions = ["happy", "surprise"]

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


def run_recognizer(clf):
    training_X, training_y, test_X, test_y = make_sets()

    training_X = training_X.reshape(training_X.shape[0], 350 * 350)
    test_X = test_X.reshape(test_X.shape[0], 350 * 350)

    clf = clf.fit(training_X, training_y)

    predict = clf.predict(test_X)

    score = clf.score(test_X, test_y)

    cm = confusion_matrix(test_y, predict, labels=[0, 1, 2, 3, 4, 5, 6, 7])

    accuracy = accuracy_score(predict, test_y)

    param_dist = {"max_depth" :  [3, 4, 5, 6, 7],
                  "max_features" : sp_randint(2, 100),
                  "min_samples_split" : sp_randint(8, 30),
                  "min_samples_leaf" : sp_randint(5, 20),
                  "n_estimators" : [100, 200, 300, 400, 500, 600],
                  "bootstrap" : [True, False],
                  "criterion" : ["gini", "entropy"]
                  }

    grid = RandomizedSearchCV(estimator=clf, param_distributions = param_dist, n_iter=100)
    grid.fit(training_X, training_y)

    y_true, y_pred = test_y, grid.predict(test_X)

    bs = grid.best_score_
    bp = grid.best_params_
    cr = classification_report(y_true, y_pred)  # isso também mostra o precision, recall e f1-score

    return accuracy, score, cm, bs, bp, cr

clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)

for i in range(0, 3):
    correct, score, cm, bs, bp, cr = run_recognizer(clf)

    print('\033[1m Etapa: ' + '\033[0m{}\n'.format(i + 1))

    print ('\033[1m Score: ' + '\033[1;34m {} \n'.format(score) + '\033[0m')

    print('\033[1m Matriz de confusao:\n\n' + '\033[0m {} \n'.format(cm))

    print('\033[1m Best Score: ' + '\033[1;34m {}\n'.format(bs) + '\033[0m')

    print('\033[1m Best Params:\n\n' + '\033[0m {}\n'.format(bp))

    print('\033[1m Classification Report:\n\n' + '\033[0m {}'.format(cr))

    metascore.append(correct)

metascore = np.array(metascore)

print("\n\033[1m Acuracy: \033[1;34m %1.2f%% \033[0m" % (metascore.mean() * 100))
print("\033[1m Margin of Error for More and Less: \033[1;34m %1.2f%%" % (metascore.std() * 100))
