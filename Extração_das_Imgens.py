'''
Este algoritmo é reposável por separar em pastas as imagens de acordo com o rotulo definido
na base de dados. Ex: Se uma imagem possui o rotulo 7, ela será classificada com surpresa e
será colocada na pasta supresa.
'''

import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] # emoções
participants = glob.glob("base\\source_emotion\\*") # retorna uma lista com os caminhos das pastas

for x in participants: #itera sobre as pastas de source_emotion
    part = "%s" % x[-4:]
    for sessions in glob.glob("{}\\*" % x):# itera sobre as subpastas de source emotion
        for files in glob.glob("{}\\*" % sessions): # itera sobre cada imagem na subpasta
            current_session = files[25:-30]

            file = open(files, 'r')

            emotion = int(float(file.readline()))

            sourcefile_emotion = glob.glob("base\\source_images\\{}\\{}\\*".format(part, current_session))[-1] # pega a ultima imagem, a que possui a emoção bem definida
            sourcefile_neutral = glob.glob("base\source_images\\{}\\{}\\*" % (part, current_session))[0] # pega a primeira imagem, neutra
            
            # As variaveis acima guardam o caminho das imagem obtidas, logo abaixo o mesmo procedimento
            
            dest_neut = "base\sorted_set\\neutral\\{}" % sourcefile_neutral[28:] # local para armazenamento da imagem neutra
            dest_emot = "base\sorted_set\\{}\\{}" % (emotions[emotion], sourcefile_emotion[28:]) #local para armazenamento da imagem da emoção bem definida

            copyfile(sourcefile_neutral, dest_neut) # Realiza copia para o destino
            copyfile(sourcefile_emotion, dest_emot)
