# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 00:33:48 2019

@author: chris
"""

import os
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from math import floor
import cv2
from keras.models import Model
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, AveragePooling2D, GaussianNoise, GlobalAveragePooling2D,concatenate,merge,Input, MaxPooling2D, Add
from keras.layers import Reshape, UpSampling2D, Activation, Dropout, Flatten, Conv2DTranspose
from keras.models import model_from_json, Sequential,load_model


def noise(n): #noice
    return np.random.uniform(-1.0, 1.0, size = [n,100])
    #return np.random.normal(-1.0, 1.0, size = [n,100])   #常態分布

if __name__ == '__main__':
    Num = int(input("Please input how many images you want to generate:"))
    generatorinput = load_model('./Models/Gan5.h5')
    Count = 0
    noise_dim = 100
    generatorinput.summary()
    num = 0
    im2 = generatorinput.predict(noise(Num))
    
    for i in range(Num):
        A = Image.fromarray(np.uint8(im2[i]*127.5+127.5))
        A.save("./Result/output"+str(Count)+".png")
        Count += 1
      
