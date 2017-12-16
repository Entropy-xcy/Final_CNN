
# 导入框架
import os
import sys
import csv
import keras
import logging
import numpy as np
import tensorflow as tf
import keras.utils.vis_utils
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import MaxPooling2D
from ipykernel import kernelapp as app
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils, generic_utils
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten





#准备数据集
##############################
print("Loading data...")

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')

Y_train = np.load('Y_train.npy')
Y_test = np.load('Y_test.npy')


print("Data loaded.")
print("Training set shape: " + str(X_train.shape) + 
      ", " + str(X_train.shape[0]) + " samples")
print("Testing set shape: " + str(X_test.shape) + 
      ", " + str(X_test.shape[0]) + " samples")

##############################

epochs = 100
batch_size = 128

# lr动态调整


def scheduler(epochs):
    if epochs < 40:
        return 0.1
    elif epochs < 80:
        return 0.05
    elif epochs < 120:
        return 0.03
    elif epochs < 160:
        return 0.008
    return 0.001

datagen_train = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False)

datagen_train.fit(X_train)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', 
                 input_shape=(32,32,1)))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3, kernel_initializer='he_normal'))
model.add(BatchNormalization(axis=1))
model.add(Activation('softmax'))

change_lr = LearningRateScheduler(scheduler)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])
result = model.fit_generator(datagen_train.flow
                             (X_train,Y_train, batch_size = batch_size), 
                             epochs = epochs,
                             shuffle=True, 
                             callbacks=[change_lr],
                             verbose=1,
                             validation_data=(X_test, Y_test))
model.summary()


# 绘制acc及loss曲线

def plot_acc_loss(result, epochs):
    acc = result.history['acc']
    loss = result.history['loss']
    val_acc = result.history['val_acc']
    val_loss = result.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(epochs), acc, label='Train_acc')
    plt.plot(range(epochs), val_acc, label='Test_acc')
    plt.title('Accuracy over ' + str(epochs) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(epochs), loss, label='Train_loss')
    plt.plot(range(epochs), val_loss, label='Test_loss')
    plt.title('Accuracy over ' + str(epochs) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()
    
plot_acc_loss(result, epochs)

# 最后验证，使用生成器

steps = 2048

datagen_test = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False)

datagen_test.fit(X_test)

score = model.evaluate_generator(datagen_test.flow(X_test, Y_test),
                                 steps = steps)
print('Test loss:' , score[0])
print('Test accuracy:',score[1])

# 保存模型图及模型

plot_model(model, show_shapes=True, to_file='final.png')

model.save('Final.h5')