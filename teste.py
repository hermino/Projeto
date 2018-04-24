# -*- coding: utf-8 -*-
import cv2
import glob
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import randint as sp_randint
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

metascore1 = []
metascore2 = []
metascore3 = []
metascore4 = []

# emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
emotions = ["happy", "surprise"]


def get_param_dist(label):
    if (label == 'Random Forest'):
        param_dist = {"randomforestclassifier__max_depth": [3, 4, 5, 6, 7],
                      "randomforestclassifier__max_features": [.1, .25, .5, .75, 1.],
                      "randomforestclassifier__max_leaf_nodes": sp_randint(2, 100),
                      "randomforestclassifier__min_samples_split": sp_randint(8, 30),
                      "randomforestclassifier__min_samples_leaf": sp_randint(5, 20),
                      "randomforestclassifier__n_estimators": [100, 200, 300, 400, 500, 600],
                      "randomforestclassifier__bootstrap": [True, False],
                      "randomforestclassifier__n_jobs": [5, 20, 50, 100, 200],
                      "randomforestclassifier__criterion": ["gini", "entropy"],
                      "pca__iterated_power": ["auto"],
                      "pca__svd_solver": ["full", "randomized"],
                      "pca__whiten": [True, False],
                      "pca__random_state": sp_randint(2, 20),
                      "pca__n_components": [200, 500, 700, 1000, 2000]
                      }
    elif(label == 'Extra Trees'):

        param_dist = {"extratreesclassifier__n_estimators": [100, 200, 300, 400, 500, 600],
                      "pca__whiten": [True, False],
                      "extratreesclassifier__max_features": [.1, .25, .5, .75, 1.],
                      "extratreesclassifier__min_samples_split": sp_randint(8, 30),
                      "pca__iterated_power": ["auto"],
                      "pca__random_state": sp_randint(2, 20),
                      "pca__svd_solver": ["full", "randomized"],
                      "extratreesclassifier__bootstrap": [True, False],
                      "extratreesclassifier__n_jobs": [5, 20, 50, 100, 200],
                      "extratreesclassifier__max_leaf_nodes": sp_randint(2, 100),
                      "extratreesclassifier__criterion": ["gini", "entropy"],
                      "pca__n_components": [200, 500, 700, 1000, 2000],
                      "extratreesclassifier__min_samples_leaf": sp_randint(5, 20),
                      "extratreesclassifier__max_depth": [3, 4, 5, 6, 7]
                      }
    elif(label == 'Gradient Boosting'):

        param_dist = {"pca__whiten": [True, False],
                      "gradientboostingclassifier__min_samples_leaf": sp_randint(5, 20),
                      "gradientboostingclassifier__criterion": [],
                      "gradientboostingclassifier__max_leaf_nodes": sp_randint(2, 100),
                      "pca__iterated_power": ["auto"],
                      "gradientboostingclassifier__max_features": [.1, .25, .5, .75, 1.],
                      "gradientboostingclassifier__n_estimators": [100, 200, 300, 400, 500, 600],
                      "pca__svd_solver": ["full", "randomized"],
                      "pca__n_components": [200, 500, 700, 1000, 2000],
                      "gradientboostingclassifier__max_depth": [3, 4, 5, 6, 7]
                      }
    else:
        param_dist = {}

    return param_dist

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


def run_recognizer(clf, label):
    training_X, training_y, test_X, test_y = make_sets()

    training_X = training_X.reshape(training_X.shape[0], 350 * 350)
    test_X = test_X.reshape(test_X.shape[0], 350 * 350)

    clf = clf.fit(training_X, training_y)

    predict = clf.predict(test_X)

    score = clf.score(test_X, test_y)

    cm = confusion_matrix(test_y, predict, labels=[0, 1, 2, 3, 4, 5, 6, 7])

    accuracy = accuracy_score(predict, test_y)

    print(clf.get_params().keys())

    param_dist = get_param_dist(label)

    grid = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, n_iter=100)
    grid.fit(training_X, training_y)

    y_true, y_pred = test_y, grid.predict(test_X)

    bs = grid.best_score_
    bp = grid.best_params_
    cr = classification_report(y_true, y_pred)  # isso também mostra o precision, recall e f1-score

    return accuracy, score, cm, bs, bp, cr


clf1 = make_pipeline(PCA(copy=True,
                         iterated_power='auto',
                         n_components=2000,
                         random_state=sp_randint(2, 20),
                         svd_solver='full',
                         whiten=False
                         ),
                     RandomForestClassifier(n_estimators=700, n_jobs=-1)
                     )

clf2 = make_pipeline(PCA(copy=True,
                         iterated_power='auto',
                         n_components=2000,
                         random_state=sp_randint(2, 20),
                         svd_solver='full',
                         whiten=False
                         ),
                     ExtraTreesClassifier(n_estimators=500)
                     )

clf3 = make_pipeline(PCA(copy=True,
                         iterated_power='auto',
                         n_components=2000,
                         random_state=sp_randint(2, 20),
                         svd_solver='full',
                         whiten=False
                         ),
                     GradientBoostingClassifier(n_estimators=500)
                     )

eclf = VotingClassifier(estimators=[('rf', clf1), ('et', clf2), ('gb', clf3)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'Voting']):

    for i in range(0, 3):
        correct, score, cm, bs, bp, cr = run_recognizer(clf, label)

        print('\033[1m Etapa: ' + '\033[0m{}\n'.format(i + 1))

        print ('\033[1m Score: ' + '\033[1;34m {} \n'.format(score) + '\033[0m')

        print('\033[1m Matriz de confusao:\n\n' + '\033[0m {} \n'.format(cm))

        print('\033[1m Best Score: ' + '\033[1;34m {}\n'.format(bs) + '\033[0m')

        print('\033[1m Best Params:\n\n' + '\033[0m {}\n'.format(bp))

        print('\033[1m Classification Report:\n\n' + '\033[0m {}'.format(cr))

        if(label == 'Random Forest'):
            metascore1.append(correct)
        elif(label == 'Extra Trees'):
            metascore2.append(correct)
        elif(label == 'Gradient Boosting'):
            metascore3.append(correct)
        else:
            metascore4.append(correct)

metascore1 = np.array(metascore1)
metascore2 = np.array(metascore2)
metascore3 = np.array(metascore3)
metascore4 = np.array(metascore4)

print("\n\033[1m Acuracy the \033[1;31m Random Forest\033[1m: \033[1;34m %1.2f%% \033[0m" % (metascore1.mean() * 100))
print("\033[1m Margin of Error for More and Less: \033[1;34m %1.2f%% \n" % (metascore1.std() * 100))

print("\n\033[1m Acuracy the \033[1;31m Extra Trees\033[1m: \033[1;34m %1.2f%% \033[0m" % (metascore2.mean() * 100))
print("\033[1m Margin of Error for More and Less: \033[1;34m %1.2f%% \n" % (metascore2.std() * 100))

print("\n\033[1m Acuracy the \033[1;31m Gradient Boosting\033[1m: \033[1;34m %1.2f%% \033[0m" % (metascore3.mean() * 100))
print("\033[1m Margin of Error for More and Less: \033[1;34m %1.2f%% \n" % (metascore3.std() * 100))

print("\n\033[1m Acuracy the \033[1;31m Voting\033[1m: \033[1;34m %1.2f%% \033[0m" % (metascore4.mean() * 100))
print("\033[1m Margin of Error for More and Less: \033[1;34m %1.2f%% \n" % (metascore4.std() * 100))
