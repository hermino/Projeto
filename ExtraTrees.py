import cv2
import glob
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier

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

    training_X = training_X.reshape(training_X.shape[0], 350*350)
    test_X = test_X.reshape(test_X.shape[0], 350*350)

    clf = clf.fit(training_X, training_y)

    predict = clf.predict(test_X)

    score = clf.score(test_X, test_y)

    cm = confusion_matrix(test_y, predict, labels=[0, 1, 2, 3, 4, 5, 6, 7])

    accuracy = accuracy_score(predict, test_y)

    return accuracy, score, cm

clf = ExtraTreesClassifier(n_estimators=500)


for i in range(0, 3):

    correct, score, cm = run_recognizer(clf)

    print("Etapa {}".format(i+1))

    print("Score: {}".format(score))

    print("Matriz de confusao:")
    print(cm)

    metascore.append(correct)

metascore = np.array(metascore)

print("\n\033[1m Acuracy: \033[1;34m %1.2f%% \033[0m" % (metascore.mean() * 100))
print("\033[1m Margin of Error for More and Less: %1.2f%% \n" % (metascore.std() * 100))
