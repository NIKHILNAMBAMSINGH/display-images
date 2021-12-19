# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:57:15 2020

@author: sonia aribam
"""

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
import timeit

import warnings
warnings.filterwarnings('ignore')

batch_size = 32
num_classes = 4
epochs = 50
# input image dimensi=ons
img_rows, img_cols = 224, 224

train_gen=ImageDataGenerator(rescale=1./255, horizontal_flip=True)
val_gen=ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_path='C:/Users/Dell/Desktop/c/train_data'
validation_path='C:/Users/Dell/Desktop/c/val_data'
test_path='C:/Users/Dell/Desktop/c/test_data'

train_data=train_gen.flow_from_directory(train_path,target_size=(img_rows,img_cols),batch_size=batch_size,class_mode='categorical')
val_data=train_gen.flow_from_directory(validation_path,target_size=(img_rows,img_cols),batch_size=batch_size,class_mode='categorical')

input_shape=(img_rows,img_cols,3)

model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (7, 7), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(128, (7, 7), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(4, activation='sigmoid'))
model.summary()

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history=model.fit_generator(train_data, steps_per_epoch=30,epochs=epochs,validation_data=val_data,validation_steps=34)

train_labels=train_data.classes
print(train_data.class_indices)
print(train_labels)


print(history.history.keys())

val_data.class_indices

import matplotlib.pyplot as plt


plt.plot(history.history['loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['val_loss'])
plt.title('validation loss')
plt.xlabel('epoch')
plt.legend(['validation'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper left')
plt.show()


plt.plot(history.history['val_acc'])
plt.title('validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['validation'],loc='upper left')
plt.show()


test_data=ImageDataGenerator(rescale=1./255, horizontal_flip=True).flow_from_directory(test_path,
                                            target_size=(img_rows,img_cols),
                                            batch_size = batch_size,
                                            class_mode='categorical',
                                            shuffle=False)

test_labels=test_data.classes

print(test_data.class_indices)
print(test_labels)



from sklearn.metrics import classification_report, confusion_matrix

num_of_test_samples = 320
Y_pred = model.predict_generator(test_data, num_of_test_samples //batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_data.classes, y_pred))
print('Classification Report')

target_names = ['Canine', 'Incisor','molar','premolar']
print(classification_report(test_data.classes, y_pred, target_names=target_names))



#to check how much are predicted correct

count = 0
j=0
for i in test_labels:
    if i==y_pred[j]:
        count+=1
    j+=1

print(count," correct out of ",j)
