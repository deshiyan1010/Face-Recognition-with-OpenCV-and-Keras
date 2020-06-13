import cv2
import numpy as np 

def collect(face_cascade_path = r'D:\Vinayak\Face Rec Flask\xmls\lbpcascade_frontalface.xml'):
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    cap = cv2. VideoCapture(0)
    data = []
    label = []
    name = []
    labelizer = -1
    counter = 1
    while True:
        _,img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        k = cv2.waitKey(10)

        try:
            x,y,w,h = faces[0]
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            cv2.imshow("frame",img)
            
            if k==ord("o"):
                data.append(gray[y:y+h,x:x+w])
                label.append(labelizer)
                counter+=1
                print("No. of sameple collected for subject {} is {}".format(labelizer,counter))

            if k==ord("n"):
                name.append(input(str("Enter your name")))
                counter = 0 
                labelizer+=1
                data.append(gray[y:y+w,x:x+h])
                label.append(labelizer)
                print("No. of sameple collected for subject {} is {}".format(labelizer,counter))

            if k==ord("d"):
                return data,label,name
                break

            np.save("face.npy",data)
            np.save("label.npy",label)
            np.save("name.npy",name)

            
            

        except Exception as e:
            cv2.imshow("frame",img)

        
        if k==27:
            return data,label,name
            break
    cap.release()
    cv2.destroyAllWindows()
""" if __name__=="__main__":
    collect() """