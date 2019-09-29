from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D , Conv2D
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#%%
img_rows, img_cols = 64,64
img_channels = 3

path1 = "images"
path2 = "images_resized"
#%%
listing = os.listdir(path1) 
num_samples=len(listing)
print(num_samples)

for file in listing:
    im = Image.open(path1 + '/' + file)   
    img = im.resize((img_rows,img_cols))  
    img.convert('RGB').save(path2+'/'+file,'JPEG')        
#%%
imlist = os.listdir(path2)
im1 = np.array(Image.open('images_resized' + '/'+ imlist[0]))
m,n = im1.shape[0:2]
imnbr = len(imlist)
immatrix = np.array([np.array(Image.open('images_resized' + '/'+ im2)).flatten()
              for im2 in imlist],'f')
#%%                
'''
label=np.ones((num_samples,),dtype = int)
label[0:1260]=0
label[1260:2716]=1
label[2716:]=2
'''
label = []
for i in imlist:
    if i[0] == 'p':
        label.append(0)
    elif i[2] == 'g':
        label.append(2)
    else:
        label.append(1)

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]


#plt.imshow(img)
#print (train_data[0].shape)
#print (train_data[1].shape)
#%%
batch_size = 32
nb_classes = 3
nb_epoch = 5

nb_filters = 32
nb_pool = 3
nb_conv = 5

(X, y) = (train_data[0],train_data[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#i = 256
#plt.imshow(X_train[i, 0], interpolation='nearest')
#print("label : ", Y_train[i,:])
#%%
cnn = Sequential()
cnn.add(Conv2D(filters=32, 
               kernel_size=(2,2), 
               strides=(1,1),
               padding='same',
               input_shape=(64,64,3),
               data_format='channels_last'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))
cnn.add(Conv2D(filters=16,
               kernel_size=(2,2),
               strides=(1,1),
               padding='valid'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))
cnn.add(Flatten())        
cnn.add(Dense(32))
cnn.add(Activation('relu'))
cnn.add(Dropout(0.25))
cnn.add(Dense(3))
cnn.add(Activation('softmax'))
#cnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
cnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
#            verbose=1, validation_data=(X_test, Y_test))
hist = cnn.fit(X_train, Y_train, batch_size=batch_size, epochs=30)
#%%
_,accuracy = cnn.evaluate(X_test,Y_test)
print(accuracy)
#%%
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
'''
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
'''
# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))
# COMPILE
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=50)
#%%
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print(plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
#%%
plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])

# Confusion Matrix
#%%
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
Y_pred = cnn.predict(X_test)
print(Y_pred)
#y_pred = np.argmax(Y_pred, axis=1)
y_pred = []
for i in Y_pred:
    if i[0] - i[2] > 0.05:
        y_pred.append(0)
    elif i[2] - i[0] > 0.05:
        y_pred.append(2) 
    else:
        y_pred.append(1)
print(pd.Series(y_pred).value_counts())
p=cnn.predict_proba(X_test) # to predict probability
pm = np.max(p,axis = 1)
y_test = np.argmax(Y_test,axis=1)
#%%
print(y_pred[0])
print(y_test[0])
print(pm[0])
#%%
w = 0
for i in range(len(y_pred)):
    if abs(y_pred[i] - y_test[i]) == 2:
        w = w+1
    elif abs(y_pred[i] - y_test[i]) == 1:
        w = w+1
    #elif abs(y_pred[i] - y_test[i]) == 0 and pm[i] < 0.4:
    #    w = w+1
print(w/len(y_pred))
#%%
target_names = ['class 0(Negative)', 'class 1(Neutral)', 'class 2(Positive)']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))
#%%
model.save("cnn_pred_1.h5")
