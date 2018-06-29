# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 22:35:36 2018

@author: akilj

Smile Detecor use Haar Smile Features
"""

import cv2

# Create Cascade for face

font = cv2.FONT_HERSHEY_SIMPLEX
posF = (20,50)
posS = (20,80)
fScale = 1
fColorG = (86, 239, 124)
fColorB = (98, 98, 249)
lineType = 2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray, img):
    '''
    Script to label a colored image
    '''
    
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    if len(faces) == 0:
        cv2.putText(img,'NO FACES', posF, font, fScale,fColorB,lineType)
    else:
        cv2.putText(img,'FACES FOUND', posF, font, fScale,fColorG,lineType)
    for x, y, width, height in faces:
        # create rectangle for the faces
        cv2.rectangle(img, (x,y), (x+width, y+height), (237, 73, 73), 3)
        
        # for each face detected, look for a smile
        roi_g = gray[x:x+width, y:y+height]
        roi_c = img[x:x+width, y:y+height]
        smile = smile_cascade.detectMultiScale(roi_g,1.4,20)
        eyes = eye_cascade.detectMultiScale(roi_g,1.1,3)
        
        if len(smile):
            smile = [smile[0]] # Truncates array if multiple smiles detected
            cv2.putText(img,'SMILE DETECTED', posS, font, fScale/2,fColorG,lineType)
        else:
            cv2.putText(img,'NO SMILE DETECTED', posS, font, fScale/2,fColorB,lineType)
            
        for ex, ey, ewidth, eheight in eyes:
            cv2.rectangle(roi_c, (ex, ey), (ex+ewidth, ey+eheight),(92, 94, 234), 2)

        for sx, sy, swidth, sheight in smile:
            cv2.rectangle(roi_c, (sx,sy), (sx+swidth, sy+sheight), (49, 204, 219), 2)
    
    return img

# 0 for webcam, 1 for external source
video_cap = cv2.VideoCapture(0)

while True:
    _ , frame = video_cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
    
video_cap.release()
cv2.destroyAllWindows()
            
            
            