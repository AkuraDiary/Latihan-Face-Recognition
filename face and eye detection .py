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
eyes_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')
#profile_face = cv2.CascadeClassifier('Cascades/haarcascade_profileface.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
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
    '''
    PFaces = profile_face.detectMultiScale(gray)
    for (x,y,w,h) in PFaces:
        cv2.rectangle(img,(x,y), (x+w, y+h), (255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    '''
    eyes = eyes_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
        )
    
    for (x2,y2,w2,h2) in eyes:
        #cv2.rectangle(img,(x,y), (x+w, y+h), (0,255,0),2)
        roi_gray = gray[y2:y2+h2, x2:x2+w2]
        roi_color = img[y2:y2+h2, x2:x2+w2]
        
        eye_center = ( x2 + w2//2, y2 + h2//2)
        radius = int(round((w2 + h2)*0.25))
        frame = cv2.circle(img, eye_center, radius, (0, 255, 0 ), 4)

        cv2.putText(
                    img, 
                    str("pakai masker"), 
                    (x2+5,y2-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w, y+h), (0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        smile = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25)
            )
        """eyes = eyes_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
            )
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(img, eye_center, radius, (0, 255, 0 ), 4)"""
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
        cv2.putText(
                    img, 
                    str("nggak pakai masker"), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )

    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: #press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
