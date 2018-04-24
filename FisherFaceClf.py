import cv2
import glob
import random
import numpy as np

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
#emotions = ["happy", "surprise"]
file = open('base1\\resultados.txt', 'w')


def menu():
    print("Menu de escolha do Classificador: ")
    n = int(input("Digite uma opcao de classificador: "))
    if n == 1:
        print("Classificador FisherFace: ")
        file.write("1\n")
        classifier = cv2.face.createFisherFaceRecognizer()
    elif n == 2:
        print("Classificador EigenFace: ")
        file.write("2\n")
        classifier = cv2.face.createEigenFaceRecognizer()
    elif n == 3:
        print("Classificador LBPHF: ")
        file.write("3\n")
        classifier = cv2.face.createLBPHFaceRecognizer()

    return classifier


def predicao(prediction_data, prediction_labels):
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred = classifier.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:

            cv2.imwrite("base1\difficult\\%s_%s_%s.jpg" % (emotions[prediction_labels[cnt]],
                                                     emotions[pred], cnt), image)
            incorrect += 1
            cnt += 1
    return ((100 * correct) / (correct + incorrect))


def get_files(emotion):
    files = glob.glob("base1\dataset\\%s\\*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]
    prediction = files[-int(len(files) * 0.2):]
    return training, prediction


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:

        training, prediction = get_files(emotion)

        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    #print "\nClassificador de treinamento"
    #print "O tamanho do conjunto de treinamento e: ", len(training_labels), "imagens"
    classifier.train(training_data, np.asarray(training_labels))

    #print "Previsao do conjunto de classificacao:"

    porcentagem = predicao(prediction_data, prediction_labels)

    return porcentagem


metascore = []

classifier = menu()


for i in range(0, 10):
    correct = run_recognizer()
    file.write("{} {}\n".format(i, correct))
    print("{}".format(i))
    #print "Obteve", correct, "porcento correto!"
    metascore.append(correct)

metascore = np.array(metascore)

print ('\033[1m Score: ' + '\033[1;34m {} \n'.format(metascore.mean()) + '\033[0m')
print ('\033[1m Margin of Erro: ' + '\033[1;34m {} \n'.format(metascore.std()) + '\033[0m')
file.write("{}\n".format(np.mean(metascore)))
file.write("{}\n".format(np.str(metascore)))

file.close()