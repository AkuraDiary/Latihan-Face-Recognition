import numpy as np
import cv2
'''
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1) # Flip camera vertically
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
'''
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 640) #set width
cap.set(4, 480) #set height

while (True):
    ret, img = cap.read()
    #img = cv2.flip(img, -1)
    #img = 
    #img = np.array(img, dtype=np.uint8)
    #img = cv2.imread(cap,0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#cv2.COLOR_BAYER_GR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2, 
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w, y+h), (0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: #press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
