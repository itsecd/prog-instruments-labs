# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 18:10:49 2020

@author: tsoyg
"""
import time
from pystackreg import StackReg
from skimage import io
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import numpy as np
import os
from os import listdir
from os.path import isfile, join

total_time = time.time()
pathIn = "Test Sequence"  # откуда брать файлы
pathOut = "Test Sequence Out/test5-9 - pystackreg/5. BILINEAR/"  # куда сохранять файлы
files = [f for f in listdir(pathIn) if isfile(join(pathIn, f))]  # сами картинки
if not os.path.exists(pathOut): os.makedirs(pathOut)  # создать путь, если ещё нет
ref_image = img_as_ubyte(rgb2gray(io.imread(join(pathIn, files[0]))))  # задаём эталон
io.imsave(pathOut + files[0], img_as_ubyte(ref_image))  # сохранить первый файл
sko = 0
sko_arr = [None] * (len(files) - 1)
time_arr = [None] * (len(files) - 1)
for i in range(1, len(files) - 1):
    # -------Процесс согласования НАЧАЛО------#
    start_time = time.time()
    offset_image = img_as_ubyte(rgb2gray(io.imread(join(pathIn, files[i]))))
    reg_instance = StackReg(StackReg.BILINEAR)
    corrected_image = reg_instance.register_transform(ref_image, offset_image)
    corrected_image = img_as_ubyte(
        (corrected_image - np.min(corrected_image)) / (np.max(corrected_image) - np.min(corrected_image)))
    io.imsave(pathOut + files[i], corrected_image)
    time_arr[i - 1] = time.time() - start_time
    # -------Процесс согласования КОНЕЦ-------#
    print("--- %s seconds ---" % (time.time() - start_time))
    sko_arr[i - 1] = np.sum((corrected_image.astype("float") - ref_image.astype("float")) ** 2)
    sko_arr[i - 1] /= float(ref_image.shape[0] * ref_image.shape[1])
    print("SKO[" + str(i) + "]: " + str(sko_arr[i - 1]) + "\n")
    sko += sko_arr[i - 1]
    # -------------Рассчёт СКО----------------#
sko = sko / (len(files) - 1)
sko_arr[len(files) - 2] = "Average SKO: " + str(sko)
time_arr[len(files) - 2] = "Total Time: " + str(time.time() - total_time)
with open(pathOut + 'SKO.txt', 'w') as f:
    for i in range(1, len(sko_arr) + 1):
        if (i != 50):
            f.write("SKO№%s:\t%s\tExecution time %s\n" % (i, sko_arr[i - 1], time_arr[i - 1]))
        else:
            f.write("\t%s\t%s\n" % (sko_arr[i - 1], time_arr[i - 1]))
    # ------------Запись в файл---------------#
print("Average SKO: " + str(sko))
print(str(time_arr[len(files) - 2]))
