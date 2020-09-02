import cv2
import numpy as np 
import os

from dataCollection import collect
from training import train
from predictor import predict
from boundingBox import detect

    
    
if __name__=="__main__":
    
    
    
    #os.chdir("D:\Vinayak\Face Rec Flask")

    if 'facerec.h5' not in os.listdir():
        print("No old entries train your face.")
        print("Train up to 200 images per subject.")

        face,label,name = collect()

        face_rec = train(face,label)

    inputx = str(input("Train on face? Y / anything for no: "))

    if inputx =="Y" or inputx =="y":

        print("Collecting data")
        print("Train up to 200 images per subject")

        face,label,name = collect()

        face_rec = train(face,label)

    name_lst = np.load("name.npy",allow_pickle=True)

    capx = cv2.VideoCapture(0)

    while capx.isOpened():

        _,frame = capx.read()

        try:
            img,x,y,w,h = detect(frame)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if w!=0:
                name,conf = predict(gray[y:y+h,x:x+w],name_lst)
                print(conf)
                img = cv2.putText(img,str(name),(x,y),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

            frame = img
        
        except Exception as e:
            print(e)
            pass
        img0 = cv2.resize(frame,(400,300))
        cv2.imshow("b",np.array(img0))

        k = cv2.waitKey(10)

        if k == 27:
            break

capx.release()
cv2.destroyAllWindows() 

        
        



