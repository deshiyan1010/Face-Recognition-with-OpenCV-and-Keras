import cv2
import numpy as np 
from boundingBox import detect
import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Activation
from tensorflow.keras.models import Sequential
import numpy as np
from keras.models import load_model


def label_name(label,name_lst):

    return name_lst[label]



def predict(img,name_lst):

    model=tf.keras.models.load_model('facerec.h5')
  
    size=70
    img = cv2.resize(img,(size,size))

    img=[img]
    img=np.array(img)

    #print(img.shape)
    img = img.reshape(-1,size,size,1)

    img = np.array(img)/127.0-1
    predict = model.predict(img)
    
    name = label_name(np.argmax(predict),name_lst) 
    
    return name,predict