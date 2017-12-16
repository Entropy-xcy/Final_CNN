import os
import csv
import numpy as np
from keras.utils import np_utils, generic_utils

dir_train = 'cifar_3/train'
dir_test = 'cifar_3/test'

name_train = []
name_test = []

for file in os.walk(dir_train):
    name_train.append(file)

for file in os.walk(dir_test):
    name_test.append(file)

X_train = np.array([np.array(Image.open('cifar_3/train/'+name_train[0][2][0]))])
X_test = np.array([np.array(Image.open('cifar_3/test/'+name_test[0][2][0]))])

for i in range(1,15000):
    X_train = np.append(train,[(np.array(Image.open('cifar_3/train/'+name_train[0][2][i])))],axis=0)

for i in range(1,3000):
    X_test = np.append(test,[(np.array(Image.open('cifar_3/test/'+name_test[0][2][i])))],axis=0)

X_train = (X_train.astype('float32')/255).reshape(15000,32,32,1)
X_test = (X_test.astype('float32')/255).reshape(3000,32,32,1)

train_label=[]
test_label=[]

with open('train.csv','r') as f:
    label_train = csv.reader(f)
    for row in label_train:
        train_label = np.append(train_label,row)

with open('test.csv','r') as f:
    label_test = csv.reader(f)
    for row in label_test:
        test_label = np.append(test_label,row)

train_label[0] = 2
test_label[0] = 0

Y_train = np_utils.to_categorical(train_label, 3)
Y_test = np_utils.to_categorical(test_label, 3)

np.save('X_train',X_train)
np.save('X_test',X_test)

np.save('Y_train',Y_train)
np.save('Y_test',X_test)

