import librosa
import os
from scipy.signal import spectrogram
from display import read_file

def get_data_set(dyrectory):

    files_name = os.listdir(dyrectory)
    y = []
    files_names =[]
    srs = []
    for file in files_name:
        path = dyrectory + "/" + file
        [a, b] = read_file(path)
        y.append(a)
        files_names.append(file)
        srs.append(b)

    return y, files_names, srs


def audio_to_mel(y,sr):
    return librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax = 8000)


def define_class(y1, y2):
    dec = 0
    if (("kitchen" in y1) and ("kitchen" in y2)) and (("mi" in y1) and ("mi" in y2)): dec = 1
    if (("kitchen" in y1) and ("kitchen" in y2)) and (("android" in y1) and ("android" in y2)): dec = 1
    if (("kitchen" in y1) and ("kitchen" in y2)) and (("iphone" in y1) and ("iphone" in y2)): dec = 1
    if (("room" in y1) and ("room" in y2)) and (("mi" in y1) and ("mi" in y2)): dec = 1
    if (("room" in y1) and ("room" in y2)) and (("android" in y1) and ("android" in y2)): dec = 1
    if (("room" in y1) and ("room" in y2)) and (("iphone" in y1) and ("iphone" in y2)): dec = 1
    if (("electro" in y1) and ("electro" in y2)): dec = 1
    if (("second_pink" in y1) and ("second_pink" in y2)): dec = 1

    return dec

def spectrogramm(x):
    f1, t1, s1 = spectrogram(x)
    return f1, t1, s1