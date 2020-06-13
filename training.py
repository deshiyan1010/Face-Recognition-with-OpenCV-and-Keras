import cv2
import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Activation
from tensorflow.keras.models import Sequential

def train(face = np.load('face.npy',allow_pickle=True),label = np.load('label.npy',allow_pickle=True)):

    for i,img in enumerate(face):  
        face[i]=cv2.resize(img,(70,70))

    size=70
    x_train,y_train = [x for x in face],label
    x_train = np.array(x_train)/127.0-1
    x_train = x_train.reshape(-1,size,size,1)
    y_train = np.array(y_train)

    model = Sequential()

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  

    model.add(Dense(64))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    tf.keras.backend.set_floatx('float64')

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.3)

    model.save("facerec.h5")

    return model

if __name__=="__main__":

    face = np.load("face.npy",allow_pickle=True)
    label = np.load("label.npy",allow_pickle=True)

    train(face,label)