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
from keras.models import model_from_json, Sequential
from keras.optimizers import Adam,RMSprop


print("Importing Images...")
Count = 0
Images = []
#BaseData = 'C:\\Users\\chris\\Desktop\\DCGAN256'
images_path = './images'
print(images_path)
files = os.listdir(images_path)


for filename in files:
    temp1 = cv2.imread(os.path.join(images_path,filename))
    temp1 = cv2.resize(temp1,(256,256),interpolation=cv2.INTER_AREA)
    #print(filename)
    temp = temp1[:, :, ::-1]
    Images.append((temp-127.5) / 127.5)
    #Images.append(np.flip(Images[-1], 1))

print(len(Images))

def zero():  #Zero label，label smoothing
    return np.random.uniform(0.0, 0.05, size = [1])
    #return np.random.normal(0.0, 0.05, size = [1])        #常態分布 

def one():   #one label，label smoothing
    return np.random.uniform(0.95, 1.0, size = [1])
    #return np.random.normal(0.95, 1.0, size = [1])        #常態分布
def noise(n): #noice
    return np.random.uniform(-1.0, 1.0, size = [n,100])
    #return np.random.normal(-1.0, 1.0, size = [n,100])   #常態分布

def residual_block(x, filters, kernel_size):  #Discriminator residual block

    X = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding='same', activation='relu')(x)
    X = BatchNormalization()(X)

    X = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding='same', activation='relu')(X)
    X = BatchNormalization()(X)

    X = Add()([X, x])
    return X

def residual_blockUp(x, filters, kernel_size): #Generator residual block

    X = BatchNormalization(momentum = 0.5)(x)
    X = LeakyReLU()(X)
    X = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides=(1,1),padding = 'same')(X)
    X = BatchNormalization(momentum = 0.5)(x)
    X = LeakyReLU()(X)
    X = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides=(1,1),padding = 'same')(X)

    X = Add()([X, x])
    return X

def generator():
    noise_shape = (100,)
    inputs = Input(shape=noise_shape)
    x = Dense(64*16*16, input_shape=noise_shape,activation='relu')(inputs)
    x = Reshape((16,16,64))(x)
    x = BatchNormalization(momentum=0.5)(x)

    
    #Second Layer
    x = UpSampling2D()(x)  # 6x8 -> 12x16
    x = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding='same',activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)


    x = UpSampling2D()(x)  # 6x8 -> 12x16
    x = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding='same',activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = UpSampling2D()(x)  # 6x8 -> 12x16
    x = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding='same',activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = UpSampling2D()(x)  # 6x8 -> 12x16
    x = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding='same',activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding='same',activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)


    x = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding='same')(x)
    x = Activation('tanh')(x)

    generatoroutput = Model(inputs=inputs, outputs = x)
  
    return generatoroutput

def discriminator():
    
    inputs = Input(shape=(256,256,3))

    x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    logits = Flatten()(x)
    logits = Dense(1, activation = 'sigmoid')(logits)
    discriminatoroutput = Model(inputs=inputs, outputs = logits)
    optimizer = Adam(0.0002, 0.5)
    discriminatoroutput.compile(optimizer = optimizer, loss = 'binary_crossentropy')
    #discriminatoroutput.compile(optimizer = optimizer, loss = 'mean_squared_error')
    return discriminatoroutput


if __name__ == '__main__':
  
    discriminatorinput = discriminator()
    generatorinput = generator()
    noise_dim = 100
    # Make the discriminator untrainable when we are training the generator.  This doesn't effect the discriminator by itself
    discriminatorinput.trainable = False
    generatorinput.summary()
    discriminatorinput.summary()

    gan_input = Input(shape=(noise_dim,))
    gan_output = discriminatorinput(generatorinput(gan_input))
    gan = Model(gan_input,gan_output)
    gan.summary()
    #optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
    optimizer = Adam(0.0002, 0.5)
    gan.compile(optimizer = optimizer, loss = 'binary_crossentropy')
    #gan.compile(optimizer = optimizer, loss = 'mean_squared_error')
    # Training loop
    iteration = 10000
    batch_size = 8
    start = 0
    num = 0
    print("Start Training!!")
    for step in range(iteration):  
        random_vectors = noise(batch_size)  #加入隨機噪音   
        generator_images = generatorinput.predict(random_vectors)   #使用generator輸出隨機噪音生成圖片

        stop = start + batch_size        
        real_img = Images[start:stop]
        labelreal_data = []                 #真實資料標籤
        labelfalse_data = []                #假資料標籤
        labelfakeTrue_data = []             #生成出來的資料標籤，假裝是1
        for i in range(stop-start):
            labelreal_data.append(one())
            labelfalse_data.append(zero())
            labelfakeTrue_data.append(one())

        d_loss_real = discriminatorinput.train_on_batch(np.array(real_img),np.array(labelreal_data))  #計算真實圖片的loss
        d_loss_fake = discriminatorinput.train_on_batch(np.array(generator_images),np.array(labelfalse_data))  #計算假圖片的loss

        random_vectors = noise(batch_size)
        Generator_loss = gan.train_on_batch(np.array(random_vectors),np.array(labelfakeTrue_data))    #計算GAN的整體loss

        start += batch_size
        if start > len(Images) - batch_size:
            start = 0
        
        print("Step : "+str(step)+", D_fake_loss : "+str(d_loss_fake)+", D_real_loss : "+str(d_loss_real)+", G_loss : "+str(Generator_loss))
        #print("Step : %d, D_fake_loss : %f, D_real_loss : %f, G_loss : %f"%(step,d_loss_fake,d_loss_real,Generator_loss))
        if step % 500 == 0:
            im2 = generatorinput.predict(noise(25))
        
            r1 = np.concatenate(im2[:5], axis = 1)
            r2 = np.concatenate(im2[5:10], axis = 1)
            r3 = np.concatenate(im2[10:15], axis = 1)
            r4 = np.concatenate(im2[15:20], axis = 1)
            r5 = np.concatenate(im2[20:25], axis = 1)
            #r6 = np.concatenate(im2[40:48], axis = 1)
        
            c1 = np.concatenate([r1, r2, r3, r4, r5], axis = 0)
      
            x = Image.fromarray(np.uint8(c1*127.5+127.5))
            x.save("./Result/i"+str(num)+".png")
            generatorinput.save('./Models/Gan'+str(num)+'.h5')
            num += 1
