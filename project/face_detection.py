# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 13:35:46 2020

@author: vamshikrishna Bandari
"""
import cv2
 
faceCascade= cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")

img = cv2.imread('1.jpg')


imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

 
faces = faceCascade.detectMultiScale(imgGray,1.1,4)
 
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)

img = cv2.resize(img,(400,600))
cv2.imshow("Result", img)
cv2.waitKey(0)