# Libraries
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from zipfile import ZipFile
from keras import layers
from glob import glob

import warnings
warnings.filterwarnings('ignore')


# Constants & Hyperparameters
BATCH_SIZE = 5
EPOCHS = 15
IMG_SIZE = 300
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)



## ZipFile Extraction
#with ZipFile('deepfake and real images.zip') as data:
#    data.extractall()


# Data Preprocessing
X_train = []
Y_train = []
X_test = []
Y_test = []

train_data_path = 'Dataset/Train'
test_data_path = 'Dataset/Test'

classes = os.listdir(train_data_path)

for i, name in enumerate(classes):
    images = glob(f'{train_data_path}/{name}/*.jpg')

    for image in images:
        img = cv2.imread(image)

        X_train.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y_train.append(i)

for i, name in enumerate(classes):
    images = glob(f'{test_data_path}/{name}/*.jpg')

    for image in images:
        img = cv2.imread(image)

        X_test.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y_test.append(i)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

Y_train = pd.get_dummies(Y_train)
Y_test = pd.get_dummies(Y_test)


# Model Callbacks
Checkpoint = ModelCheckpoint('output/model.h5',
                             monitor= 'val_accuracy',
                             verbose= 1,
                             save_best_only= True,
                             save_weights_only= False
                             )


# ResNet Model
base_model = keras.applications.ResNet50(include_top= False,
                                         weights= 'imagenet',
                                         input_shape= IMG_SHAPE,
                                         pooling= 'max'
                                         )

model = keras.Sequential(
    base_model,

    layers.Dropout(0.1),

    layers.Dense(128, activation= 'relu'),

    layers.Dense(64, activation= 'relu'),

    layers.Dropout(0.1),

    layers.Dense(2, activation= 'softmax')
)

model.compile(optimizer= 'adam',
              loss= 'binary_crossentropy',
              metrics = ['accuracy']
              )


# Model Training
model.fit(X_train, Y_train,
          batch_size= BATCH_SIZE,
          callbacks= Checkpoint,
          verbose= 1,
          validation_data= (X_test, Y_test)
          )