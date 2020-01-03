

import numpy as np
import cv2

#We point OpenCV's cascade classifier function to where our classifier (XML file format is stored)
face_classifier = cv2.CascadeClassifier("D:/Python/Python datasets/New folder/xml/classifier/cascade.xml")

#Load our image then convert it to grayscale
image = cv2.imread("D:/Python/Python datasets/New folder/images/number.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Our classifier returns the ROI(Region of interest) of the detected face as tuple
#It stores the top left coordinate and the bottom right coordinate
faces = face_classifier.detectMultiScale(gray,1.2,5)

#When no faces detected, face_classifier return the empty tuple
if faces is ():
    print("No faces found")
    
# we iterate through our faces array and draw a rectangle
# over each face in faces
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('Face Detection',image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
