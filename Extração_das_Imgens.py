import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
participants = glob.glob("base\\source_emotion\\*")

for x in participants:
    part = "%s" % x[-4:]
    for sessions in glob.glob("%s\\*" % x):
        for files in glob.glob("%s\\*" % sessions):
            current_session = files[25:-30]

            file = open(files, 'r')

            emotion = int(float(file.readline()))

            sourcefile_emotion = glob.glob("base\\source_images\\{}\\{}\\*".format(part, current_session))[-1]
            sourcefile_neutral = glob.glob("base\source_images\\%s\\%s\\*" % (part, current_session))[0]

            dest_neut = "base\sorted_set\\neutral\\%s" % sourcefile_neutral[28:]
            dest_emot = "base\sorted_set\\%s\\%s" % (emotions[emotion], sourcefile_emotion[28:])

            copyfile(sourcefile_neutral, dest_neut)
            copyfile(sourcefile_emotion, dest_emot)