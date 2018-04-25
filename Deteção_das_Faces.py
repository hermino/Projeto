'''
Este código realiza a identificação da face e corte da mesma.
A partir daqui os dados ja estão prontos para o treino de classificadores.
'''

import cv2
import glob

# Iniciar o Haar Cascade que é tecnica de identificação de objetos, mais informações aqui: 
# https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html
# Poderia ser só uma Cascade

faceDet = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] # Emoçõoes

def detect_faces(emotion):
    files = glob.glob("base\\sorted_set\\{}\\*".format(emotion))# Itera sobre as pasta de cada emoção

    filenumber = 0
    i = 1
    for f in files:
        frame = cv2.imread(f) # ler a imagem
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Transforma em escala de cinza
        
        # Detecta a face
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE) 
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        # Adicina a face a uma lista
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""

         # Itera sobre a lista de faces
        for (x, y, w, h,) in facefeatures:
            gray = gray[y:y + h, x:x + w]
            try:
                out = cv2.resize(gray, (350, 350)) # Realiza o corte, ele despreza todo o resto deixando apena o rosto
                cv2.imwrite("base\\dataset\\%s\\%s.jpg" % (emotion, filenumber), out) # Renomeia e salva
            except:
                pass
        i = i + 1
        filenumber += 1


for emotion in emotions:
    detect_faces(emotion)
    print()
