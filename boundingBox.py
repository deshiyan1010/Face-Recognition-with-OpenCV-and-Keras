import cv2
import numpy as np 
def detect(img,face_cascade_path = r'D:\Vinayak\Face Rec Flask\xmls\lbpcascade_frontalface.xml'):
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    try:
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for x,y,w,h in faces:
            #val = predict(img[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #img = cv2.putText(img,str(val),(x,y),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            return img,x,y,w,h
    except Exception as e:
        print(e)
        pass
    return img,0,0,0,0

    


if __name__=="__main__":
    """ face = np.load("face.npy",allow_pickle=True)
    cv2.imshow("x",face[1])
    cv2.waitKey(0)
    img,x,y,w,h = detect(face[1])
    cv2.imshow("x",img)
    print(x,y,w,h)
    cv2.waitKey(0) """
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        _,frame = cap.read()

        img,x,y,w,h = detect(frame)

        cv2.imshow("x",img)

        k=cv2.waitKey(10)

        if k==27:
            break